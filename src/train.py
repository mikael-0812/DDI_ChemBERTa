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

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ndlong/mm-dti"
subfolder = "ChemBERTa"

model = SmilesOnlyDDIModel(
    model_name=model_name,
    lambda_supcon=0.1,
    subfolder=subfolder,
    num_classes=86,
    freeze_encoder=True,
).to(device)

tokenizer = model.tokenizer

def load_data_model(filepath, shuffle, batch_size=128):
    dataset = DDIDataset(filepath)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_ddi(b, tokenizer, max_length=128, device=device),
    )
    return dataset, loader

train_dataset, train_loader = load_data_model('dataset/inductive_data/train_f.csv', shuffle=True)
test_dataset, test_loader = load_data_model('dataset/inductive_data/test_f.csv', shuffle=False)

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


# ============================================================
# TRAIN WITH CHECKPOINT + TEST AFTER EACH EPOCH
# ============================================================

def train_and_evaluate(model, train_loader, test_loader, num_epochs=5, use_amp=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scaler)

    for epoch in range(start_epoch, num_epochs):

        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
        model.train()

        total_loss = 0.0
        batch_times = []

        import time
        epoch_start = time.time()

        # ========== TRAIN LOOP ==========
        for batch_idx, (enc1, enc2, enc_neg1, enc_neg2, labels) in enumerate(train_loader):

            batch_start = time.time()

            optimizer.zero_grad()
            logits_pos, loss_pos = model(enc1, enc2, labels)
            logits_neg, _ = model(enc_neg1, enc_neg2, labels=None)

            pos_scores = logits_pos[torch.arange(labels.size(0)), labels]  # (B,)
            neg_scores = logits_neg[torch.arange(labels.size(0)), labels]  # (B,)

            margin = 0.5
            rank_loss = F.relu(margin - (pos_scores - neg_scores)).mean()

            total_loss = loss_pos + 0.5 * rank_loss  # λ_rank = 0.5

            total_loss.backward()
            optimizer.step()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Train batch {batch_idx + 1}/{len(train_loader)} "
                      f"- loss: {total_loss.item():.4f} - time: {batch_times[-1]:.3f}s")

        epoch_end = time.time()

        avg_batch = sum(batch_times) / len(batch_times)
        print(f"\n=== Epoch {epoch + 1} DONE ===")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Epoch Time: {epoch_end - epoch_start:.2f}s")
        print(f"Batch Time Avg: {avg_batch:.3f}s")

        # ========== TEST AFTER EVERY EPOCH ==========
        print("\nEvaluating on TEST set...")
        avg_loss, acc, macro_f1, macro_auc, macro_aupr = evaluate(model, test_loader)

        # ========== SAVE BEST MODEL ==========
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New BEST model saved! Macro-F1 = {best_f1:.4f}")

        # ========== SAVE CHECKPOINT TO RESUME LATER ==========
        save_checkpoint(epoch, model, optimizer, scaler, best_f1)

def evaluate(model, loader):
    model.eval()

    all_logits = []
    all_labels = []

    total_loss = 0
    total_batches = len(loader)

    import time
    test_start = time.time()

    with torch.no_grad():
        for enc1, enc2, labels in loader:
            logits, _ = model(enc1, enc2, labels=None)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Gộp toàn bộ (không theo batch)
    all_logits = torch.cat(all_logits, dim=0).numpy()        # (N, C)
    all_labels = torch.cat(all_labels, dim=0).numpy()        # (N,)

    preds = all_logits.argmax(axis=1)

    # ============================
    # 1. ACC, Macro F1
    # ============================
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro")

    # ============================
    # 2. Macro AUC + Macro AUPR an toàn
    #   Không WARNING, không NaN
    # ============================
    macro_auc_list = []
    macro_aupr_list = []

    num_classes = all_logits.shape[1]

    for c in range(num_classes):
        y_true = (all_labels == c).astype(int)    # one-vs-rest
        y_score = all_logits[:, c]

        # Skip nếu class này không có cả positive và negative
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        # AUC class c
        try:
            auc_c = roc_auc_score(y_true, y_score)
            macro_auc_list.append(auc_c)
        except ValueError:
            pass

        # AUPR class c
        try:
            aupr_c = average_precision_score(y_true, y_score)
            macro_aupr_list.append(aupr_c)
        except ValueError:
            pass

    # Nếu tất cả lớp đều bị skip → trả về 0 thay cho NaN
    macro_auc = np.mean(macro_auc_list) if len(macro_auc_list) > 0 else 0
    macro_aupr = np.mean(macro_aupr_list) if len(macro_aupr_list) > 0 else 0

    # ============================
    # 3. Tổng hợp
    # ============================
    avg_loss = total_loss / total_batches
    test_time = time.time() - test_start

    print("\n========== TEST RESULTS (clean no-warning) ==========")
    print(f"  Test Loss     : {avg_loss:.4f}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Macro AUC     : {macro_auc:.4f}")      # guaranteed safe
    print(f"  Macro AUPR    : {macro_aupr:.4f}")     # guaranteed safe
    print(f"  Test time     : {test_time:.2f}s")
    print("=====================================================\n")

    return avg_loss, acc, macro_f1, macro_auc, macro_aupr



if __name__ == "__main__":
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=5,
        use_amp=True
    )

