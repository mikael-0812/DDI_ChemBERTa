import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import SmilesOnlyDDIModel, DDIDataset, collate_ddi
import os
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, average_precision_score
)
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


drug_dict = pd.read_csv("dataset/drugbank/drug_smiles.csv")
id2smiles = dict(zip(drug_dict['drug_id'], drug_dict['smiles']))

model_name = "ndlong/mm-dti"
subfolder_smiles = "ChemBERTa"
subfolder_unimol = "Uni-Mol"

model = SmilesOnlyDDIModel(
    model_name=model_name,
    lambda_supcon=0.1,
    subfolder=subfolder_smiles,
    num_classes=86,
    freeze_encoder=True,
).to(device)

# uni_mol_extract = 3D_Conformer(
#
# )

tokenizer = model.tokenizer

def load_data_model(filepath, shuffle, is_test, batch_size=128):
    dataset = DDIDataset(filepath, id2smiles)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_ddi(b, tokenizer, is_test, device=device, max_length=128),
    )
    return dataset, loader

train_dataset, train_loader = load_data_model('dataset/inductive_data/fold0/train.csv', shuffle=True, is_test=False)
test_dataset, test_loader = load_data_model('dataset/inductive_data/fold0/s2.csv', shuffle=False, is_test=True)


CHECKPOINT_PATH = "checkpoint_ddi.pt"
BEST_MODEL_PATH = "best_ddi_model.pt"


def save_checkpoint(epoch, model, optimizer, scaler, best_f1):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_f1": best_f1,
    }
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"Saved checkpoint at epoch {epoch + 1}")


def load_checkpoint(model, optimizer=None, scaler=None):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found → start training from scratch.")
        return 0, 0.0

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_f1 = ckpt["best_f1"]

    print(f"Loaded checkpoint → resume from epoch {start_epoch + 1}")
    return start_epoch, best_f1

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    n_samples = 0

    all_logits = []
    all_labels = []

    test_start = time.time()

    for enc1, enc2, labels in loader:
        logits, _ = model(enc1, enc2, labels=None)

        # ====== CE LOSS ======
        ce_loss = F.cross_entropy(logits, labels)
        total_loss += ce_loss.item() * labels.size(0)
        n_samples += labels.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    # ====== Compute avg loss ======
    avg_loss = total_loss / n_samples

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    preds = all_logits.argmax(axis=1)

    # ACC & F1
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro")

    # AUC & AUPR
    macro_auc_list = []
    macro_aupr_list = []

    num_classes = all_logits.shape[1]
    for c in range(num_classes):
        y_true = (all_labels == c).astype(int)
        y_score = all_logits[:, c]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        try:
            macro_auc_list.append(roc_auc_score(y_true, y_score))
        except:
            pass

        try:
            macro_aupr_list.append(average_precision_score(y_true, y_score))
        except:
            pass

    macro_auc = np.mean(macro_auc_list) if macro_auc_list else 0
    macro_aupr = np.mean(macro_aupr_list) if macro_aupr_list else 0

    print("\n========== TEST RESULTS ==========")
    print(f"  Test CE Loss : {avg_loss:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Macro AUC    : {macro_auc:.4f}")
    print(f"  Macro AUPR   : {macro_aupr:.4f}")
    print(f"  Test time    : {time.time() - test_start:.2f}s")
    print("===================================\n")

    return avg_loss, acc, macro_f1, macro_auc, macro_aupr


def train_and_evaluate(model, train_loader, test_loader, num_epochs=5, use_amp=True):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-4
    )
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scaler)

    for epoch in range(start_epoch, num_epochs):
        print(f"\n========== EPOCH {epoch + 1}/{num_epochs} ==========")
        model.train()

        total_loss = 0.0
        batch_times = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch_t0 = time.time()

            enc1, enc2, enc_neg1, enc_neg2, labels = batch
            enc1 = enc1.to(device)
            enc2 = enc2.to(device)
            enc_neg1 = enc_neg1.to(device)
            enc_neg2 = enc_neg2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ======== forward + loss (có AMP) ========
            with autocast(enabled=(scaler is not None)):
                # positive pair
                logits_pos, loss_pos = model(enc1, enc2, labels)

                # negative pair (không cần loss trong model, chỉ cần logits)
                logits_neg, _ = model(enc_neg1, enc_neg2, labels=None)

                # ranking loss
                pos_scores = logits_pos[torch.arange(labels.size(0)), labels]
                neg_scores = logits_neg[torch.arange(labels.size(0)), labels]

                margin = 0.5
                rank_loss = F.relu(margin - (pos_scores - neg_scores)).mean()

                batch_loss = loss_pos + 0.5 * rank_loss  # λ_rank = 0.5

            # ======== backward + step ========
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

            total_loss += batch_loss.item()
            batch_times.append(time.time() - batch_t0)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {batch_loss.item():.4f} "
                    f"Time: {batch_times[-1]:.3f}s"
                )

        epoch_time = time.time() - epoch_start
        avg_batch_time = sum(batch_times) / max(len(batch_times), 1)

        print(f"\n[EPOCH {epoch + 1}] Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"[EPOCH {epoch + 1}] Epoch Time: {epoch_time:.2f}s "
              f"(Avg batch: {avg_batch_time:.3f}s)")

        # ======== EVALUATE SAU MỖI EPOCH ========
        print("\n>>> Evaluating on TEST set...")
        _, _, macro_f1, macro_auc, macro_aupr = evaluate(model, test_loader)

        # ======== LƯU BEST MODEL =========
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[BEST] New best model saved! Macro-F1 = {best_f1:.4f}")

        # ======== LƯU CHECKPOINT =========
        save_checkpoint(epoch, model, optimizer, scaler, best_f1)

if __name__ == "__main__":
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=5,
        use_amp=True
    )

