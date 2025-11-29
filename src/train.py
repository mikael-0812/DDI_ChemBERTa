import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import SmilesOnlyDDIModel, DDIDataset, collate_ddi
from sklearn.metrics import accuracy_score, f1_score


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ndlong/mm-dti"
subfolder = "ChemBERTa"

model = SmilesOnlyDDIModel(
    model_name=model_name,
    subfolder=subfolder,
    num_classes=86,
    freeze_encoder=True,
    lambda_supcon=0.1,
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


def train(train_dataset, train_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    import time
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        epoch_start = time.time()
        batch_times = []

        for batch_idx, (enc1, enc2, labels) in enumerate(train_loader):

            batch_start = time.time()

            optimizer.zero_grad()
            logits, loss = model(enc1, enc2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            # In progress mỗi 50 batch (tùy chọn)
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} "
                      f"- loss: {loss.item():.4f} "
                      f"- time: {batch_times[-1]:.3f}s")

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Tổng hợp thống kê batch time
        avg_batch = sum(batch_times) / len(batch_times)
        min_batch = min(batch_times)
        max_batch = max(batch_times)

        print(f"\n===== Epoch {epoch+1}/{num_epochs} completed =====")
        print(f"  ➤ Loss avg: {total_loss/len(train_loader):.4f}")
        print(f"  ➤ Epoch time: {epoch_time:.2f} seconds")
        print(f"  ➤ Batch time (avg/min/max): "
              f"{avg_batch:.3f}s / {min_batch:.3f}s / {max_batch:.3f}s")
        print("==============================================\n")


def evaluate(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0
    total_batches = len(test_loader)

    import time
    test_start = time.time()

    with torch.no_grad():
        batch_times = []

        for batch_idx, (enc1, enc2, labels) in enumerate(test_loader):

            batch_start = time.time()

            logits, _ = model(enc1, enc2, labels=None)  # không tính SupCon
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            # Print mỗi vài batch
            if (batch_idx + 1) % 1000 == 0:
                print(f"  Test batch {batch_idx+1}/{total_batches} "
                      f"- loss: {loss.item():.4f} "
                      f"- time: {batch_times[-1]:.3f}s")

    test_end = time.time()
    test_time = test_end - test_start

    # Gộp kết quả
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    avg_loss = total_loss / total_batches
    avg_batch = sum(batch_times) / len(batch_times)
    min_batch = min(batch_times)
    max_batch = max(batch_times)

    print("\n========== TEST RESULTS ==========")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print("---------------------------------")
    print(f"  Test time: {test_time:.2f} seconds")
    print(f"  Batch time (avg/min/max): "
          f"{avg_batch:.3f}s / {min_batch:.3f}s / {max_batch:.3f}s")
    print("=================================\n")

    return avg_loss, acc, f1

if __name__ == "__main__":

    train(train_dataset, train_loader)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
