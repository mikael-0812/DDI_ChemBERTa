import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import SmilesOnlyDDIModel, DDIDataset, collate_ddi
import os
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

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

def load_data_model(filepath, shuffle, batch_size=8):
    dataset = DDIDataset(filepath)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_ddi(b, tokenizer, max_length=128, device=device),
    )
    return dataset, loader

train_dataset, train_loader = load_data_model('../dataset/drugbank/train_f.csv', shuffle=True)
test_dataset, test_loader = load_data_model('../dataset/drugbank/test_f.csv', shuffle=False)

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
        for batch_idx, (enc1, enc2, labels) in enumerate(train_loader):

            batch_start = time.time()

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    logits, loss = model(enc1, enc2, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(enc1, enc2, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            if (batch_idx + 1) % 1000 == 0:
                print(f"  Train batch {batch_idx + 1}/{len(train_loader)} "
                      f"- loss: {loss.item():.4f} - time: {batch_times[-1]:.3f}s")

        epoch_end = time.time()

        avg_batch = sum(batch_times) / len(batch_times)
        print(f"\n=== Epoch {epoch + 1} DONE ===")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Epoch Time: {epoch_end - epoch_start:.2f}s")
        print(f"Batch Time Avg: {avg_batch:.3f}s")

        # ========== TEST AFTER EVERY EPOCH ==========
        print("\nEvaluating on TEST set...")
        loss_test, acc_test, f1_test = evaluate(model, test_loader)

        # ========== SAVE BEST MODEL ==========
        if f1_test > best_f1:
            best_f1 = f1_test
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New BEST model saved! Macro-F1 = {best_f1:.4f}")

        # ========== SAVE CHECKPOINT TO RESUME LATER ==========
        save_checkpoint(epoch, model, optimizer, scaler, best_f1)


def evaluate(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    total_loss = 0.0
    total_batches = len(test_loader)

    import time
    test_start = time.time()

    with torch.no_grad():
        batch_times = []

        for batch_idx, (enc1, enc2, labels) in enumerate(test_loader):

            batch_start = time.time()

            logits, _ = model(enc1, enc2, labels=None)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            if (batch_idx + 1) % 1000 == 0:
                print(f"  Test batch {batch_idx+1}/{total_batches} "
                      f"- loss: {loss.item():.4f} "
                      f"- time: {batch_times[-1]:.3f}s")

    test_end = time.time()
    test_time = test_end - test_start

    # Gộp kết quả
    all_preds = torch.cat(all_preds).numpy()              # (N,)
    all_labels = torch.cat(all_labels).numpy()            # (N,)
    all_logits = torch.cat(all_logits).numpy()            # (N, C)

    # ========= ACC & Macro-F1 ========= #
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    num_classes = all_logits.shape[1]
    labels_onehot = torch.nn.functional.one_hot(
        torch.tensor(all_labels),
        num_classes=num_classes
    ).numpy()

    # macro AUC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(labels_onehot, all_logits, average="macro", multi_class="ovr")
    except:
        auc_macro = float("nan")

    # ========= Multi-class AUPR ========= #
    try:
        aupr_macro = average_precision_score(labels_onehot, all_logits, average="macro")
    except:
        aupr_macro = float("nan")

    avg_loss = total_loss / total_batches
    avg_batch = sum(batch_times) / len(batch_times)
    min_batch = min(batch_times)
    max_batch = max(batch_times)

    print("\n========== TEST RESULTS ==========")
    print(f"  Test Loss     : {avg_loss:.4f}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Macro AUC     : {auc_macro:.4f}")
    print(f"  Macro AUPR    : {aupr_macro:.4f}")
    print("---------------------------------")
    print(f"  Test time     : {test_time:.2f} seconds")
    print(f"  Batch time    : {avg_batch:.3f}s (min={min_batch:.3f}s, max={max_batch:.3f}s)")
    print("=================================\n")

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "auc_macro": auc_macro,
        "aupr_macro": aupr_macro,
    }


if __name__ == "__main__":
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=5,
        use_amp=True
    )

