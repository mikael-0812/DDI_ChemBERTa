import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, RobertaModel
from typing import List, Optional

class DDIDataset(Dataset):
    """
    Tạo cặp negative sample :
    dataset : d1 - d2 - type - split($h : head and $t : tail)
    Get smile for each drug by dictionary ('id_to_smile')
    """
    def __init__(self, csv_path, id_to_smiles: dict):
        df = pd.read_csv(csv_path)

        self.d1 = df["d1"].astype(str).tolist()
        self.d2 = df["d2"].astype(str).tolist()
        self.labels = df["type"].astype(int).tolist()
        self.neg_raw = df["split"].astype(str).tolist()

        self.id_to_smiles = id_to_smiles  # dict: DrugBankID -> SMILES

    def __len__(self):
        return len(self.d1)

    def __getitem__(self, idx):

        drug1 = self.d1[idx]
        drug2 = self.d2[idx]
        label = self.labels[idx]
        neg_field = self.neg_raw[idx]

        neg_id, flag = neg_field.split("$")
        flag = flag.lower()

        if flag == "h":   # head : drug1
            neg_pair_1 = self.id_to_smiles[drug1]
            neg_pair_2 = self.id_to_smiles[neg_id]
        else:             # flag == "t" -> tail : drug2
            neg_pair_1 = self.id_to_smiles[neg_id]
            neg_pair_2 = self.id_to_smiles[drug2]

        return {
            "smiles1": self.id_to_smiles[drug1],
            "smiles2": self.id_to_smiles[drug2],
            "label": label,

            # negative pair (left drug, negative drug)
            "neg1": neg_pair_1,
            "neg2": neg_pair_2,
        }


def collate_ddi(batch, tokenizer, is_test, device, max_length=256):
    """
    :param batch:
    :param tokenizer: chemBERTa pretrained
    :param max_length: 256
    :param device: cpu/gpu
    :return: tokenizer
    """
    smi1 = [b["smiles1"] for b in batch]
    smi2 = [b["smiles2"] for b in batch]

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long).to(device)

    enc1 = tokenizer(smi1, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt").to(device)
    enc2 = tokenizer(smi2, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt").to(device)
    if is_test:
        return enc1, enc2, labels

    neg1 = [b["neg1"]    for b in batch]
    neg2 = [b["neg2"]    for b in batch]

    enc_neg1 = tokenizer(neg1, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
    enc_neg2 = tokenizer(neg2, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)

    return enc1, enc2, enc_neg1, enc_neg2, labels



class CrossAttentionBlock(nn.Module):
    """
    Cross_Attention nhận Q, K, V từ drug x và drug y
    Với x, y lần lượt đưa ra x(Q) và y(K, V)
    Hx <-> Hy
    Hy <-> Hx

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
        ) #Feed Forward
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
            key_padding_mask = (kv_mask == 0)  # Nếu kv_mask = 1 (không cần padding) ngược lại padding nếu bằng 0

        attn_out, attn_weights = self.attn(
            query=q, # (B, Lq, D)
            key=kv, # (B, Lk, D)
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # (B, num_heads, Lq, Lk) : trọng số attention
        )
        x = self.ln1(q + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, attn_weights


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
    labels = labels.view(-1, 1) #Chỉ so khớp smiles, nếu thêm graph embedding cần tạo cho smile x và graph x
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

class Get3DConformer(nn.Module):
    """
    Get embedding conformer from pretrained model Uni-Mol
    """
    def __init__(
        self,
        nodel_name: str,
        num_class: int = 86,
        subfolder: Optional[str] = None,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()


class SmilesOnlyDDIModel(nn.Module):
    """

    """
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
                p.requires_grad = False #đóng băng encoder, chỉ train cross-attn + classifier + SupCon

        self.cross_ab = CrossAttentionBlock(d_model, n_heads, dropout) #Hx_att
        self.cross_ba = CrossAttentionBlock(d_model, n_heads, dropout) #Hy_att

        #input vector : [z1, z2, z1*z2, |z1−z2|]
        #MLP (Linear -> GELU -> Drop -> Linear) to predict 86 classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

        self.lambda_supcon = lambda_supcon #Điều chỉnh CE loss + SupCon
        self.d_model = d_model

    def mean_pool(self, x, mask):
        """
        x: (B, L, D); mask: (B, L)
        """
        mask = mask.unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6) #Đưa Hx, Hy qua mean_pooling

    def forward(self, enc1, enc2, labels=None):
        #Encode SMILES
        out1 = self.encoder(**enc1)
        out2 = self.encoder(**enc2)

        H1 = out1.last_hidden_state  # (B, L1, D) : output layer cuối
        H2 = out2.last_hidden_state  # (B, L2, D)
        m1 = enc1["attention_mask"]  # (B, L1) : mask cho padding (đánh dấu padding)
        m2 = enc2["attention_mask"]  # (B, L2)

        #Cross-attention hai chiều
        H1_att, attn_ab = self.cross_ab(H1, H2, kv_mask=m2)  # (B, heads, L1, L2)
        H2_att, attn_ba = self.cross_ba(H2, H1, kv_mask=m1)  # (B, heads, L2, L1)

        #Pooling cho CE loss
        z1 = self.mean_pool(H1_att, m1)  # (B, D)
        z2 = self.mean_pool(H2_att, m2)

        prod = z1 * z2
        diff = torch.abs(z1 - z2)
        z_pair = torch.cat([z1, z2, prod, diff], dim=-1)  #input vector : [z1, z2, z1*z2, |z1−z2|]

        logits = self.classifier(z_pair) #multi_class prediction

        if labels is None:
            return logits, torch.tensor(0.0, device=logits.device)

        # CE loss
        ce_loss = F.cross_entropy(logits, labels)

        k = 16 #top 16 substructure drug

        all_motifs = []
        all_motif_labels = [] #tạo motif cho các substructure

        B = labels.size(0)

        for b in range(B):
            # valid token masks
            valid1 = m1[b].bool()
            valid2 = m2[b].bool()

            # attention sample b-th
            attn_ab_b = attn_ab[b]  # (heads, L1, L2)
            attn_ba_b = attn_ba[b]  # (heads, L2, L1)

            # compute importance per token
            imp1 = attn_ab_b.mean(dim=0).sum(dim=-1)  # (L1,)
            imp2 = attn_ba_b.mean(dim=0).sum(dim=-1)  # (L2,)

            imp1 = imp1[valid1]
            imp2 = imp2[valid2]

            # get k-indices in the original H1_att
            idx1 = torch.topk(imp1, min(k, imp1.size(0))).indices
            idx2 = torch.topk(imp2, min(k, imp2.size(0))).indices

            h1_tokens = H1_att[b][valid1][idx1]  # (<=k, D)
            h2_tokens = H2_att[b][valid2][idx2]

            all_motifs.append(h1_tokens)
            all_motifs.append(h2_tokens)

            lab = labels[b].repeat(h1_tokens.size(0) + h2_tokens.size(0)) #Tạo nhãn cho các token của 2 thuốc
            all_motif_labels.append(lab) # Nhận tất cả index của substructure

        all_motifs = torch.cat(all_motifs, dim=0) #M = tổng số motifs của toàn batch, D = embedding dimension
        all_motif_labels = torch.cat(all_motif_labels, dim=0) #Nhãn cho từng motif

        supcon = supervised_contrastive_loss(
            all_motifs, all_motif_labels, temperature=0.1
        )

        total_loss = ce_loss + self.lambda_supcon * supcon
        return logits, total_loss

