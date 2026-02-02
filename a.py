import os
import re
import zipfile
import math
import random
from collections import Counter

import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

# ============================================================
# 0) 复现性设置：固定随机种子，保证每次运行结果尽量一致
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 自动选择GPU或CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1) 下载并读取数据集（UCI SMS Spam Collection）
# 数据集格式：每行 "label \t text"
# label: ham(正常) / spam(垃圾)
# ============================================================
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "./data_sms"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
RAW_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")


def download_and_extract():
    """若本地没有数据集，则从UCI下载并解压。"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # RAW_PATH不存在说明未解压/未下载
    if not os.path.exists(RAW_PATH):
        # zip不存在就下载
        if not os.path.exists(ZIP_PATH):
            print(f"Downloading: {UCI_ZIP_URL}")
            r = requests.get(UCI_ZIP_URL, timeout=60)
            r.raise_for_status()
            with open(ZIP_PATH, "wb") as f:
                f.write(r.content)

        # 解压到 DATA_DIR
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)

    # 确认文件存在
    assert os.path.exists(RAW_PATH), "Dataset file not found after extraction."
    print("Dataset ready:", RAW_PATH)


def load_sms():
    """
    读取短信文本与标签。
    返回：
      texts: list[str]
      labels: list[int]  1=spam, 0=ham
    """
    texts, labels = [], []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # 每行: label \t text
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            lab, txt = parts
            labels.append(1 if lab == "spam" else 0)
            texts.append(txt)
    return texts, labels


# ============================================================
# 2) 简单分词 + 构建词表（vocab）
# 这里不用BERT分词，为了作业可控、易解释
# ============================================================

# 简单正则：保留字母/数字/撇号（英文短信常见）
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str):
    """简单tokenizer：转小写 + 正则切词。"""
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq=2, max_size=20000,
                special_tokens=("<pad>", "<unk>")):
    """
    根据训练集构建词表，避免测试集信息泄漏。
    参数：
      min_freq: 低于该频次的词丢弃
      max_size: 词表最大大小
      special_tokens: <pad>填充, <unk>未知词
    返回：
      vocab: dict[str,int]
    """
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # 先放入特殊token
    vocab = {tok: i for i, tok in enumerate(special_tokens)}

    # 按词频从高到低加入
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def numericalize(text, vocab):
    """把文本转成词id序列。未知词用 <unk>。"""
    unk_id = vocab["<unk>"]
    return [vocab.get(tok, unk_id) for tok in simple_tokenize(text)]


# ============================================================
# 3) Dataset 与 Collate：负责 padding 与 attention mask
# ============================================================
class SmsDataset(Dataset):
    """
    把文本序列化并截断到max_len。
    注意：这里只存储list[int]，真正pad在collate里完成。
    """
    def __init__(self, texts, labels, vocab, max_len=128):
        self.labels = labels
        self.seqs = []
        for t in texts:
            ids = numericalize(t, vocab)[:max_len]  # 截断
            self.seqs.append(ids)
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_batch(batch, pad_id):
    """
    将一个batch的不同长度序列pad到同一长度，并返回mask。
    返回：
      x: (B, T) LongTensor  pad后的token id
      attn_mask: (B, T) BoolTensor  True表示padding位置（给Transformer用）
      y: (B,) LongTensor  标签
    """
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)

    # x初始化为pad
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)

    # key_padding_mask: True表示需要mask（padding）
    attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)

    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn_mask[i, len(s):] = True  # 后面补pad的部分都mask掉

    y = torch.tensor(labels, dtype=torch.long)
    return x, attn_mask, y


# ============================================================
# 4) Transformer模型：Embedding + 位置编码 + Encoder + 池化 + 分类器
# ============================================================
class PositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码（Attention is all you need）
    生成 (1, max_len, d_model) 的pe，前向时与embedding相加。
    """
    def __init__(self, d_model: int, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1) -> 0..max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term对应 1 / (10000^(2i/d_model)) 的等价写法
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维用sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维用cos
        if d_model % 2 == 1:
            # 若d_model是奇数，cos部分维度少1，需要截断div_term
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer：不作为参数训练，但会随模型保存/加载 & 搬到GPU
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]  # 截取到序列长度T
        return self.dropout(x)


class TransformerTextClassifier(nn.Module):
    """
    使用 TransformerEncoder 做文本分类：
      input_ids -> embedding -> positional -> encoder -> mean pooling -> classifier
    """
    def __init__(
        self, vocab_size: int, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=0.1, num_classes=2, pad_id=0
    ):
        super().__init__()
        self.pad_id = pad_id

        # (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=512)

        # PyTorch原生TransformerEncoderLayer
        # batch_first=True -> 输入输出为 (B, T, D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 分类器：把 pooled 表示映射到2类
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids, key_padding_mask):
        """
        input_ids: (B, T)
        key_padding_mask: (B, T)  True=padding位置
        """
        # 1) 词嵌入
        x = self.embedding(input_ids)             # (B, T, D)

        # 2) 加位置编码
        x = self.pos_enc(x)                       # (B, T, D)

        # 3) Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        # 4) 池化：对非pad位置做mean pooling，得到句向量
        non_pad = (~key_padding_mask).float()     # (B, T)  非pad为1
        lengths = non_pad.sum(dim=1).clamp(min=1) # (B,)  每句有效长度，避免除0
        pooled = (x * non_pad.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)  # (B, D)

        # 5) 分类输出 logits
        logits = self.classifier(pooled)          # (B, 2)
        return logits


# ============================================================
# 5) 训练与评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    """
    测试/验证过程：
      - model.eval()
      - 关闭梯度 torch.no_grad()
      - 计算 acc / precision / recall / f1 / auc / confusion matrix
    """
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for x, pad_mask, y in loader:
        x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)

        logits = model(x, pad_mask)  # (B, 2)

        # spam类别的概率（softmax后取第1类）
        prob = torch.softmax(logits, dim=-1)[:, 1]  # (B,)

        # 预测类别（0/1）
        pred = torch.argmax(logits, dim=-1)         # (B,)

        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_prob.extend(prob.cpu().tolist())

    acc = accuracy_score(all_y, all_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        all_y, all_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_y, all_pred)

    # AUC需要概率分数（all_prob）
    auc = roc_auc_score(all_y, all_prob)
    return acc, p, r, f1, auc, cm


def train_one_epoch(model, loader, optimizer, criterion):
    """
    单个epoch的训练过程：
      - model.train()
      - 正向 -> loss -> 反向传播 -> 参数更新
    """
    model.train()
    total_loss = 0.0

    for x, pad_mask, y in loader:
        x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, pad_mask)          # (B, 2)
        loss = criterion(logits, y)          # cross entropy

        loss.backward()                      # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
        optimizer.step()                     # 更新参数

        total_loss += loss.item() * x.size(0)

    # 返回该epoch平均loss
    return total_loss / len(loader.dataset)


# ============================================================
# 6) 主函数：准备数据 -> 训练 -> 每轮在测试集评估 -> 保存最佳模型
# ============================================================
def main():
    # 1) 数据准备
    download_and_extract()
    texts, labels = load_sms()

    # 2) 分层划分训练集/测试集（保持spam比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 3) 只用训练集构建词表（避免测试集信息泄漏）
    vocab = build_vocab(X_train, min_freq=2, max_size=20000)
    pad_id = vocab["<pad>"]

    # 4) Dataset / DataLoader
    train_ds = SmsDataset(X_train, y_train, vocab, max_len=128)
    test_ds  = SmsDataset(X_test,  y_test,  vocab, max_len=128)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id)
    )

    # 5) 构建模型
    model = TransformerTextClassifier(
        vocab_size=len(vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=2,
        pad_id=pad_id
    ).to(DEVICE)

    # 6) 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 7) 训练并在每个epoch结束后用测试集评估
    best_f1 = 0.0
    for epoch in range(1, 11):
        # 训练（仅训练集）
        loss = train_one_epoch(model, train_loader, optimizer, criterion)

        # 测试评估（仅测试集；不更新参数）
        acc, p, r, f1, auc, cm = evaluate(model, test_loader)

        print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.4f} | "
              f"P={p:.4f} R={r:.4f} F1={f1:.4f} | AUC={auc:.4f}")
        print("Confusion Matrix:\n", cm)

        # 以F1作为“最佳模型”选择标准
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "vocab": vocab}, "best_sms_transformer.pt")

    print("Best F1:", best_f1)
    print("Saved to best_sms_transformer.pt")


if __name__ == "__main__":
    main()
