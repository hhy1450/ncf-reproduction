import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate


def train_one_epoch(model, users, items, labels, optimizer, criterion, device, batch_size):
    model.train()
    total_loss, n_batches = 0, 0
    n = len(users)
    indices = np.arange(n)
    np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]

        u_batch = torch.tensor(users[idx], device=device)
        i_batch = torch.tensor(items[idx], device=device)
        l_batch = torch.tensor(labels[idx], device=device)

        optimizer.zero_grad()
        preds = model(u_batch, i_batch)
        loss = criterion(preds, l_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train(
    model,
    train_users,
    train_items,
    train_labels,
    val_dict,
    test_dict,
    train_user_items,
    num_items,
    epochs=20,
    batch_size=256,
    lr=0.001,
    device="cpu",
    eval_k=10,
    early_stop_patience=5,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_hr, best_epoch = 0, 0
    best_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, K: {eval_k}")
    print(f"Training samples: {len(train_users)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_users, train_items, train_labels,
            optimizer, criterion, device, batch_size,
        )

        val_hr, val_ndcg = evaluate(
            model, val_dict, train_user_items, num_items, k=eval_k
        )

        print(
            f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
            f"Val HR@{eval_k}: {val_hr:.4f} | Val NDCG@{eval_k}: {val_ndcg:.4f}"
        )

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model and evaluate on test set
    model.load_state_dict(best_state)
    model = model.to(device)
    test_hr, test_ndcg = evaluate(
        model, test_dict, train_user_items, num_items, k=eval_k
    )

    print(f"\n{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test HR@{eval_k}: {test_hr:.4f} | Test NDCG@{eval_k}: {test_ndcg:.4f}")
    print(f"{'='*60}\n")

    return test_hr, test_ndcg, best_epoch
