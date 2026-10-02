"""Fetch unpaired ColabFold-server MSAs for the unchanged public domain example."""

import argparse
import hashlib
import json
import re
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = "https://api.colabfold.com"


def request_json(path, data=None):
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(SERVER + path, data=body, headers={"User-Agent": "structure-pilot/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main(ticket_id, input_json=None, output=None, cache_dir=None):
    input_json = input_json or ROOT / "inputs" / "6kzw_domain_complex.json"
    output = output or ROOT / "inputs" / "6zkw_domain_msa.a3m"
    info = json.loads(input_json.read_text())
    chains = info["chain_sequences"]
    if list(chains) != ["A", "B", "C", "D"]:
        raise ValueError("Expected alpha/beta/peptide/MHC chain order A/B/C/D")
    searched = [c for c in chains if len(chains[c]) >= 20]
    query_ids = {c: 101 + i for i, c in enumerate(searched)}
    query = "".join(f">{query_ids[c]}\n{chains[c]}\n" for c in searched)
    cache = cache_dir or ROOT / "cache" / "msa"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "out.tar.gz"
    manifest_path = ROOT / "inputs" / "msa_manifest.json" if output.name == "6zkw_domain_msa.a3m" else output.with_suffix(".manifest.json")
    ticket_path = cache / "ticket.json"
    query_sha256 = hashlib.sha256(query.encode()).hexdigest()
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if archive.exists() and previous:
        if previous.get("source_query_sha256") != query_sha256:
            raise ValueError("Cached MSA archive belongs to different input chains")
        if previous.get("archive_sha256") != hashlib.sha256(archive.read_bytes()).hexdigest():
            raise ValueError("Cached MSA archive checksum mismatch")
        ticket_id = ticket_id or previous.get("ticket_id")
    if not archive.exists():
        if ticket_id is None and ticket_path.exists():
            saved_ticket = json.loads(ticket_path.read_text())
            if saved_ticket["query_sha256"] != query_sha256:
                raise ValueError("Cached MSA ticket belongs to different queries")
            ticket_id = saved_ticket["ticket_id"]
        if ticket_id is None:
            answer = request_json("/ticket/msa", {"q": query, "mode": "env"})
            if answer["status"] in ["ERROR", "MAINTENANCE", "RATELIMIT"]:
                raise RuntimeError(answer)
            ticket_id = answer["id"]
            ticket_path.write_text(json.dumps({"ticket_id": ticket_id, "query_sha256": query_sha256}) + "\n")
        started = time.time()
        while True:
            answer = request_json("/ticket/" + ticket_id)
            print(json.dumps(answer), flush=True)
            if answer["status"] == "COMPLETE":
                break
            if answer["status"] not in ["PENDING", "RUNNING", "UNKNOWN"] or time.time() - started > 1800:
                raise RuntimeError(answer)
            time.sleep(10)
        request = urllib.request.Request(SERVER + "/result/download/" + ticket_id)
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(50_000_001)
        if len(data) > 50_000_000:
            raise ValueError("Unexpected MSA archive size")
        archive.write_bytes(data)
    by_query = {qid: [] for qid in query_ids.values()}
    with tarfile.open(archive) as handle:
        for name in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]:
            member = handle.getmember(name)
            if not member.isfile() or member.size > 100_000_000:
                raise ValueError("Unexpected archive member")
            # Read only the requested members; never extract untrusted paths.
            text = handle.extractfile(member).read().decode()
            current, reset = None, True
            for line in text.splitlines(keepends=True):
                if "\0" in line:
                    line, reset = line.replace("\0", ""), True
                if line.startswith(">") and reset:
                    current, reset = int(line[1:].strip()), False
                if current is not None and line.strip():
                    by_query[current].append(line)
    individual = {}
    counts = {}
    for chain, seq in chains.items():
        lines = "".join(by_query[query_ids[chain]]).splitlines() if chain in query_ids else [">peptide_query", seq]
        if len(lines) % 2 or lines[1] != seq:
            raise ValueError(f"MSA query sequence mismatch for {chain}")
        rows = list(zip(lines[::2], lines[1::2]))
        if any(not h.startswith(">") or len(re.sub(r"[a-z]", "", s)) != len(seq) for h, s in rows):
            raise ValueError(f"Invalid alignment for {chain}")
        individual[chain] = rows
        counts[chain] = len(rows)
    lengths = [len(s) for s in chains.values()]
    lines = ["#" + ",".join(map(str, lengths)) + "\t1,1,1,1", ">query_A\tquery_B\tquery_C\tquery_D", "".join(chains.values())]
    for i, chain in enumerate(chains):
        for header, aligned in individual[chain]:
            padded = "-" * sum(lengths[:i]) + aligned + "-" * sum(lengths[i + 1:])
            lines.extend([header, padded])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", newline="\n")
    manifest = {
        "server": SERVER, "ticket_id": ticket_id, "mode": "env", "templates": False,
        "homolog_pairing": False, "query_ids": query_ids, "rows_per_chain": counts,
        "peptide_msa": "query only, not sent to MSA server", "public_input_pdb": info["pdb"],
        "source_query_sha256": query_sha256,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "a3m_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket-id", default=None)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    main(args.ticket_id, args.input_json, args.output, args.cache_dir)
