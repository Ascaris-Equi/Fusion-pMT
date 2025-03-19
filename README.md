# Fusion-pMT
Fusion-pMT: A revolutionary T-cell immunogenicity prediction model leveraging state-of-the-art biological sequence modeling.

We introduce Fusion-pMT, a novel model utilizing sequence fusion and a unified encoder enhanced with amino acid physicochemical properties for improved immunogenicity prediction.
Installation

    pip install torch numpy pandas scikit-learn scipy tqdm

    python infer.py --input input.csv

(input.csv should contain peptide, MHC, and TCR sequence data in CSV format.)
Highlights:

1.Unified encoder integrating physicochemical properties and positional embeddings.
2.Sequence-form preserved fusion via Cross Attention.
3.Pre-trained on peptide-MHC binding data for better performance.

# Contact

Feel free to open an issue or contact authors for questions.
