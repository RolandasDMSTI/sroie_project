import time
import mlflow
import mlflow.pytorch as mlflow_pytorch

import torch

from encoders import create_encoder
from data import create_loaders
from sroie_dataset import LABELS, tag2id, id2tag
from sroie_dataset import plain_sroie_dataset
from model import BiLSTMCRF
from constraints import make_bio_constraints
from train_helpers import set_seed, evaluate

# torch.backends.cudnn.enabled = False

log_test_metrics = False
use_BIO = any(label.startswith("B-") for label in LABELS)

mlflow.set_experiment("bi-lstm-crf sroie reproducible")

dataset = plain_sroie_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")
# SEED = 0
DROPOUT = 0.5
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
HIDDEN_DIM = 128
USE_HANDCRAFTED_FEATURES = False
# USE_STYLE = False
COMMENT = "No ReLU in style"
MAX_EPOCH = 300

for SEED in range(0, 10):
    for USE_STYLE in [False, True]:
        print(COMMENT)
        print(f"{DROPOUT=} {BATCH_SIZE=} {LR=} {PATIENCE=} {SEED=} {MAX_EPOCH=}" 
            f" {USE_HANDCRAFTED_FEATURES=} {USE_STYLE=}")

        set_seed(SEED)

        with mlflow.start_run():
            mlflow.set_tag("comment", COMMENT)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("dropout", DROPOUT)
            mlflow.log_param("hidden_dim", HIDDEN_DIM)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("learning_rate", LR)
            mlflow.log_param("patience", PATIENCE)
            mlflow.log_param("max_epoch", MAX_EPOCH)
            mlflow.log_param("device", device)
            mlflow.log_param("use_features", USE_HANDCRAFTED_FEATURES)
            mlflow.log_param("use_style", USE_STYLE)

            vocab, train_loader, val_loader, test_loader = create_loaders(dataset, BATCH_SIZE)
            encoder = create_encoder(vocab)

            print(f"{encoder=}")

            mlflow.log_param("encoder", str(encoder))

            constraints = make_bio_constraints(tag2id) if use_BIO else None

            model = BiLSTMCRF(
                encoder=encoder,
                hidden_dim=HIDDEN_DIM,
                num_tags=len(tag2id),
                constraints=constraints,
                dropout=DROPOUT,
                use_features=USE_HANDCRAFTED_FEATURES,
                use_style=USE_STYLE
            )
            model.to(device)

            patience = PATIENCE    
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            epoch = 0
            best_f1 = -1
            best_epoch = 0
            epochs_without_improvement = 0

            training_start = time.time()
            for epoch in range(MAX_EPOCH):
                model.train()
                total_loss = torch.tensor(0.0, device=device)

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    features = batch["features"].to(device)
                    sizes = batch["sizes"].to(device)
                    tags = batch["tags"].to(device)
                    mask = batch["mask"].to(device)
                    attention_mask = batch.get("attention_mask") # only in BERT
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    optimizer.zero_grad()

                    loss = model.loss(input_ids, tags, mask, features, sizes, attention_mask)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    optimizer.step()
                    model.apply_constraints()

                    total_loss += loss.detach()

                train_loss = (total_loss / len(train_loader)).item()
                val_precision, val_recall, val_f1 = evaluate(model, val_loader, id2tag, device)
                test_log = ""
                if log_test_metrics:
                    test_precision, test_recall, test_f1 = evaluate(model, test_loader, id2tag, device)
                    test_log = f" | test: P={test_precision:.4f} | R={test_recall:.4f} | F1={test_f1:.4f}"
                print(f"Epoch {epoch + 1:02d} | {train_loss=:.4f} | "
                    f"val: P={val_precision:.4f} | R={val_recall:.4f} | F1={val_f1:.4f}"
                    + test_log, end=""
                )

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                }, step=epoch)

                if val_f1 > best_f1:
                    print(" *", end="")
                    best_f1 = val_f1
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), "best_model.pt")
                elif val_f1 > 0:
                    epochs_without_improvement += 1
                print()

                if epochs_without_improvement >= patience:
                    print(f"Early stopping. Best F1 was {best_f1:.4f}")
                    break

            model.load_state_dict(torch.load("best_model.pt", weights_only=True))
            test_precision, test_recall, test_f1 = evaluate(model, test_loader, id2tag, device)
            print(f"Best epoch: {best_epoch:02d}"
                f" | test: P={test_precision:.4f} | R={test_recall:.4f} | F1={test_f1:.4f}"
            )

            training_time = time.time() - training_start
            per_epoch = training_time / (epoch + 1)
            print(f"Training time: {training_time:.2f}, per epoch: {per_epoch:.2f}")

            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metrics({
                "per_epoch": per_epoch,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "best_val_f1": best_f1,
            })

            mlflow_pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle",
            )
