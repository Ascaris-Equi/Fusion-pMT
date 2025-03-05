#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pMHC-TCR Interaction Predictor
用于预测肽段、MHC和TCR序列之间的相互作用。

使用方法:
    python infer.py --input your_data.csv --output predictions.csv
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

# 设置随机种子，确保结果可重复
seed = 19961231
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 全局参数设置
pep_max_len = 20   # peptide最大长度
mhc_max_len = 34   # MHC序列最大长度
tcr_max_len = 26   # TCR序列最大长度
tgt_len = pep_max_len + mhc_max_len + tcr_max_len

# Transformer模型参数
d_model = 64    # 嵌入维度
d_ff = 512      # 前馈层隐藏层维度
d_k = d_v = 64  # 注意力机制Q、K、V的维度
n_layers = 1    # 编码器/解码器层数
n_heads = 8     # 多头注意力的头数

# 检查CUDA是否可用
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#######################################
# 模型定义部分 - 与原始模型完全一致
#######################################

# 数据集类
class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, mhc_inputs, tcr_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.mhc_inputs = mhc_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.mhc_inputs[idx], self.tcr_inputs[idx], self.labels[idx]

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    """创建用于注意力机制的填充掩码"""
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)

# 注意力计算
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        # 添加LayerNorm作为类成员
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        # 线性映射并分头
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        # 添加LayerNorm作为类成员
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, property_dim=4, property_table=None):
        """
        vocab_size: 词表大小
        d_model: 最终嵌入维度
        property_dim: 理化性质维度（此处为4）
        property_table: 大小为 [vocab_size, property_dim] 的tensor，每行对应一个氨基酸的理化性质向量
        """
        super(CombinedEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.prop_linear = nn.Linear(property_dim, d_model)
        # 融合后拼接的向量维度为 2*d_model，再投影回 d_model
        self.proj = nn.Linear(2 * d_model, d_model)
        if property_table is None:
            raise ValueError("需要提供property_table")
        self.register_buffer('property_table', property_table)
        
    def forward(self, x):
        # x: [batch_size, seq_len]，每个元素为氨基酸的索引
        emb = self.emb(x)  # [batch_size, seq_len, d_model]
        # 根据索引查表获得理化性质向量，再经过线性映射
        prop = self.prop_linear(self.property_table[x])  # [batch_size, seq_len, d_model]
        combined = torch.cat([emb, prop], dim=-1)  # [batch_size, seq_len, 2*d_model]
        out = self.proj(combined)  # [batch_size, seq_len, d_model]
        return out

class Encoder(nn.Module):
    def __init__(self, property_table):
        super(Encoder, self).__init__()
        # 从property_table获取vocab_size
        vocab_size = property_table.size(0)
        # 使用CombinedEmbedding替换原始nn.Embedding
        self.src_emb = CombinedEmbedding(vocab_size, d_model, property_dim=4, property_table=property_table)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)  # 得到融合后的嵌入表示
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        
    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, combined_enc_outputs):
        # 将encoder输出加入位置编码
        dec_outputs = self.pos_emb(combined_enc_outputs.transpose(0, 1)).transpose(0, 1)
        dec_self_attn_pad_mask = get_attn_pad_mask(combined_enc_outputs[:,:,0], combined_enc_outputs[:,:,0])
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns

class Transformer(nn.Module):
    def __init__(self, property_table):
        super(Transformer, self).__init__()
        # 三个不同序列分别使用相同的编码器结构
        self.pep_encoder = Encoder(property_table)
        self.mhc_encoder = Encoder(property_table)
        self.tcr_encoder = Encoder(property_table)
        self.decoder = Decoder()
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )
        
    def forward(self, pep_inputs, mhc_inputs, tcr_inputs):
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        mhc_enc_outputs, mhc_enc_self_attns = self.mhc_encoder(mhc_inputs)
        tcr_enc_outputs, tcr_enc_self_attns = self.tcr_encoder(tcr_inputs)
        # 合并三个编码器的输出（在序列维度上拼接）
        combined_enc_outputs = torch.cat((pep_enc_outputs, mhc_enc_outputs, tcr_enc_outputs), dim=1)
        dec_outputs, dec_self_attns = self.decoder(combined_enc_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, mhc_enc_self_attns, tcr_enc_self_attns, dec_self_attns

#######################################
# 推理相关函数
#######################################

def create_vocab():
    """创建氨基酸词汇表，确保与预训练模型匹配"""
    # 原始模型使用21个元素的词汇表
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"  # 20个氨基酸加一个填充符号'-'
    # 注意: 词汇表索引从0开始
    return {aa: i for i, aa in enumerate(amino_acids)}

def assign_properties(sequence):
    """对氨基酸序列计算理化性质"""
    aromatic = set("FWY")      # 芳香性
    acidic = set("DE")         # 酸性
    basic = set("KRH")         # 碱性
    hydrophilic = set("NQSTY")  # 亲水性

    properties = []
    for aa in sequence:
        prop = {
            "aromatic": int(aa in aromatic),
            "acidic": int(aa in acidic),
            "basic": int(aa in basic),
            "hydrophilic": int(aa in hydrophilic)
        }
        properties.append(prop)
    return properties

def create_property_table(vocab):
    """创建氨基酸理化性质查找表"""
    vocab_size = len(vocab)
    property_table = torch.zeros(vocab_size, 4)
    
    # 为词汇表中每个氨基酸计算性质
    for aa, idx in vocab.items():
        props = assign_properties(aa)  # 得到列表，每个元素为一个字典（这里只有1个字符）
        if len(props) > 0:  # 确保有返回值
            prop = [props[0]["aromatic"], props[0]["acidic"], props[0]["basic"], props[0]["hydrophilic"]]
            property_table[idx] = torch.tensor(prop, dtype=torch.float)
    
    return property_table

def make_data(data, vocab):
    """将DataFrame中的序列转换为模型输入的索引张量"""
    pep_inputs, mhc_inputs, tcr_inputs = [], [], []
    
    for pep, mhc, tcr in zip(data['peptide'], data['MHC_sequence'], data['TCR_sequence']):
        # 右侧填充至固定长度
        pep = pep.ljust(pep_max_len, '-')
        mhc = mhc.ljust(mhc_max_len, '-')
        tcr = tcr.ljust(tcr_max_len, '-')
        
        # 将序列转换为索引列表，未知氨基酸替换为0
        pep_input = [[vocab.get(n, 0) for n in pep]]
        mhc_input = [[vocab.get(n, 0) for n in mhc]]
        tcr_input = [[vocab.get(n, 0) for n in tcr]]
        
        pep_inputs.extend(pep_input)
        mhc_inputs.extend(mhc_input)
        tcr_inputs.extend(tcr_input)
    
    return torch.LongTensor(pep_inputs), torch.LongTensor(mhc_inputs), torch.LongTensor(tcr_inputs)

def transfer(y_prob, threshold=0.5):
    """转换概率值为二进制标签"""
    return np.array([1 if x > threshold else 0 for x in y_prob])

def predict(model, loader):
    """使用模型进行预测"""
    model.eval()
    y_prob_list = []
    
    with torch.no_grad():
        for pep_inputs, mhc_inputs, tcr_inputs, _ in tqdm(loader, desc="预测中"):
            pep_inputs = pep_inputs.to(device)
            mhc_inputs = mhc_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            
            outputs, _, _, _, _ = model(pep_inputs, mhc_inputs, tcr_inputs)
            probs = nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy()
            y_prob_list.extend(probs)
    
    return y_prob_list

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='预测pMHC-TCR相互作用')
    parser.add_argument('--input', required=True, help='输入CSV文件，需包含peptide, MHC_sequence, TCR_sequence三列')
    parser.add_argument('--output', default='predictions.csv', help='输出CSV文件')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='正例预测阈值')
    parser.add_argument('--model', default='model_layer1_multihead8_fold2.pkl', help='预训练模型路径')
    parser.add_argument('--vocab', default='vocab_dict.npy', help='词汇表文件路径')
    args = parser.parse_args()
    
    # 检查设备
    print(f"使用设备: {device}")
    
    # 加载词汇表
    try:
        if os.path.exists(args.vocab):
            print(f"加载词汇表: {args.vocab}")
            vocab = np.load(args.vocab, allow_pickle=True).item()
        else:
            print(f"词汇表文件不存在: {args.vocab}，将创建基础词汇表")
            vocab = create_vocab()
    except Exception as e:
        print(f"加载词汇表错误: {e}")
        print("将创建基础词汇表")
        vocab = create_vocab()
    
    # 创建理化性质表
    print("创建氨基酸理化性质表...")
    property_table = create_property_table(vocab)
    
    # 加载数据
    try:
        print(f"加载数据: {args.input}")
        data = pd.read_csv(args.input)
        print(f"已加载 {len(data)} 条序列")
    except Exception as e:
        print(f"加载输入文件错误: {e}")
        return
    
    # 检查必要列是否存在
    required_columns = ['peptide', 'MHC_sequence', 'TCR_sequence']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"错误: 缺少必要列: {missing_columns}")
        print(f"必要列: {required_columns}")
        return
    
    # 准备数据
    print("准备输入数据...")
    pep_inputs, mhc_inputs, tcr_inputs = make_data(data, vocab)
    
    # 创建数据加载器
    dummy_labels = torch.zeros(len(pep_inputs))
    loader = Data.DataLoader(
        MyDataSet(pep_inputs, mhc_inputs, tcr_inputs, dummy_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # 初始化模型
    print("初始化模型...")
    model = Transformer(property_table)
    
    # 将模型移动到设备上
    model = model.to(device)
    
    # 加载预训练权重
    if os.path.exists(args.model):
        try:
            print(f"加载预训练模型: {args.model}")
            model.load_state_dict(torch.load(args.model, map_location=device))
        except Exception as e:
            print(f"加载模型错误: {e}")
            print("将使用随机初始化的模型，结果将不准确！")
    else:
        print(f"模型文件不存在: {args.model}")
        print("将使用随机初始化的模型，结果将不准确！")
    
    # 执行预测
    print("开始预测...")
    probabilities = predict(model, loader)
    
    # 生成标签
    labels = transfer(probabilities, args.threshold)
    
    # 添加结果到数据框
    result_df = data.copy()
    result_df['Score'] = probabilities
    result_df['Label'] = labels
    
    # 保存结果
    result_df.to_csv(args.output, index=False)
    print(f"预测结果已保存至: {args.output}")
    
    # 显示摘要
    positives = sum(labels)
    print(f"结果摘要: {positives} 个正例预测 (共 {len(labels)} 条序列, {positives/len(labels)*100:.2f}%)")

if __name__ == "__main__":
    main()