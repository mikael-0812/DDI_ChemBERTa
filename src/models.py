import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, RobertaModel
from typing import List, Optional


# ==========================
# Dataset skeleton
# ==========================

class DDIDataset(Dataset):
    """
    CSV format:
    SMILES1, SMILES2, Interaction_class, NegativeSampleSMILES

    Ở phiên bản này ta dùng SMILES1, SMILES2, Interaction_class.
    SMILES_NEG có thể dùng sau cho hard negative.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.smi1 = df["d1_smiles"].astype(str).tolist()
        self.smi2 = df["d2_smiles"].astype(str).tolist()
        self.labels = df["type"].astype(int).tolist()
        # Nếu muốn dùng negative sau:
        #self.smi_neg = df["NegativeSampleSMILES"].astype(str).tolist()

    def __len__(self):
        return len(self.smi1)

    def __getitem__(self, idx):
        return {
            "smiles1": self.smi1[idx],
            "smiles2": self.smi2[idx],
            "label": self.labels[idx],
            # "smiles_neg": self.smi_neg[idx],
        }


def collate_ddi(batch, tokenizer, max_length: int = 256, device: str = "cpu"):
    smiles1 = [b["smiles1"] for b in batch]
    smiles2 = [b["smiles2"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    enc1 = tokenizer(
        smiles1,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc2 = tokenizer(
        smiles2,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Move to device
    enc1 = {k: v.to(device) for k, v in enc1.items()}
    enc2 = {k: v.to(device) for k, v in enc2.items()}
    labels = labels.to(device)

    return enc1, enc2, labels


# ==========================
# Cross-Attention Block
# ==========================

class CrossAttentionBlock(nn.Module):
    """
    Một block cross-attention đơn giản:
    Q từ seq_q, K/V từ seq_kv
    Có residual + LayerNorm + FFN như Transformer encoder.
    """
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask=None):
        """
        q:  (B, Lq, D)
        kv: (B, Lk, D)
        kv_mask: (B, Lk) 1 = keep, 0 = pad
        """
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = (kv_mask == 0)  # True = pad

        attn_out, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # (B, num_heads, Lq, Lk)
        )
        x = self.ln1(q + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, attn_weights


# ==========================
# SupCon cho motif/substructure
# ==========================

def supervised_contrastive_loss(
    features: torch.Tensor,  # (M, D)
    labels: torch.Tensor,    # (M,)
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon) ở mức motif.

    features: tất cả motif embeddings (token-level) trong batch
    labels: nhãn interaction class tương ứng với motif đó
    """
    device = features.device
    features = F.normalize(features, dim=-1)  # normalize

    M = features.size(0)
    if M < 2:
        return torch.tensor(0.0, device=device)

    # Cosine similarity matrix: (M, M)
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Label mask: positives nếu cùng label, bỏ self
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # loại bỏ diagonal (self)
    mask = mask - torch.eye(M, device=device)

    # For numerical stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Exponential of logits
    exp_logits = torch.exp(logits)  # (M, M)

    # Mask self, chỉ softmax trên các mẫu khác
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Chỉ tính trên positive pairs
    pos_mask = (mask > 0)
    num_pos = pos_mask.sum()
    if num_pos.item() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (log_prob * pos_mask).sum() / (num_pos + 1e-12)

    loss = -mean_log_prob_pos
    return loss


# ==========================
# Model: SMILES-only DDI + Substructure SupCon
# ==========================

class SmilesOnlyDDIModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        lambda_supcon: float,
        num_classes: int = 86,
        subfolder: Optional[str] = None,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Load pretrained ChemBERTa (Roberta-based)
        if subfolder is not None:
            self.encoder = RobertaModel.from_pretrained(
                model_name,
                subfolder=subfolder,
                use_safetensors=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                subfolder=subfolder,
            )
        else:
            self.encoder = RobertaModel.from_pretrained(
                model_name,
                use_safetensors=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.cross_ab = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_ba = CrossAttentionBlock(d_model, n_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

        self.lambda_supcon = lambda_supcon
        self.d_model = d_model

    def mean_pool(self, x, mask):
        """
        x: (B, L, D); mask: (B, L)
        """
        mask = mask.unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, enc1, enc2, labels=None):
        #Encode SMILES
        out1 = self.encoder(**enc1)
        out2 = self.encoder(**enc2)

        H1 = out1.last_hidden_state  # (B, L1, D)
        H2 = out2.last_hidden_state  # (B, L2, D)
        m1 = enc1["attention_mask"]  # (B, L1)
        m2 = enc2["attention_mask"]  # (B, L2)

        #Cross-attention hai chiều
        H1_att, attn_ab = self.cross_ab(H1, H2, kv_mask=m2)  # (B, heads, L1, L2)
        H2_att, attn_ba = self.cross_ba(H2, H1, kv_mask=m1)  # (B, heads, L2, L1)

        #Pooling cho CE loss
        z1 = self.mean_pool(H1_att, m1)  # (B, D)
        z2 = self.mean_pool(H2_att, m2)

        prod = z1 * z2
        diff = torch.abs(z1 - z2)
        z_pair = torch.cat([z1, z2, prod, diff], dim=-1)

        logits = self.classifier(z_pair)

        if labels is None:
            return logits, torch.tensor(0.0, device=logits.device)

        # CE loss
        ce_loss = F.cross_entropy(logits, labels)

        k = 16

        all_motifs = []
        all_motif_labels = []

        B = labels.size(0)

        for b in range(B):
            # valid token masks
            valid1 = m1[b].bool()
            valid2 = m2[b].bool()

            # attention sample b
            attn_ab_b = attn_ab[b]  # (heads, L1, L2)
            attn_ba_b = attn_ba[b]  # (heads, L2, L1)

            # compute importance per token
            imp1 = attn_ab_b.mean(dim=0).sum(dim=-1)  # (L1,)
            imp2 = attn_ba_b.mean(dim=0).sum(dim=-1)  # (L2,)

            imp1 = imp1[valid1]
            imp2 = imp2[valid2]

            # get indices in the original H1_att
            idx1 = torch.topk(imp1, min(k, imp1.size(0))).indices
            idx2 = torch.topk(imp2, min(k, imp2.size(0))).indices

            h1_tokens = H1_att[b][valid1][idx1]  # (<=k, D)
            h2_tokens = H2_att[b][valid2][idx2]

            all_motifs.append(h1_tokens)
            all_motifs.append(h2_tokens)

            lab = labels[b].repeat(h1_tokens.size(0) + h2_tokens.size(0))
            all_motif_labels.append(lab)

        all_motifs = torch.cat(all_motifs, dim=0)
        all_motif_labels = torch.cat(all_motif_labels, dim=0)

        supcon = supervised_contrastive_loss(
            all_motifs, all_motif_labels, temperature=0.1
        )

        total_loss = ce_loss + self.lambda_supcon * supcon
        return logits, total_loss

