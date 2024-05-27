import logging
import datetime
import os
import torch
import itertools
from mlp import MLP
from datasets import EmbeddingDataset
from torch.optim import AdamW
from torch.utils.data import random_split
from utils import get_embedding_model
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import GaussianNLLLoss
import argparse
from torch.distributions import Normal, MultivariateNormal
import random
from sklearn.metrics import r2_score


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE

# seed everything with 42
torch.manual_seed(42)


def setup_logging(emb_model, data_dir):
    current_time = datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S-{emb_model}")

    # Create a new directory for the current log
    log_dir = os.path.join(data_dir, f"logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"log_{current_time}.txt")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")


def print_log(*args, **kwargs):
    logging.info(*args, **kwargs)
    print(*args, **kwargs)


def step(model, loss_fn, data, labels, optimizer=None, eps=1e-7, verbose=False):
    data, labels = data.to(DEVICE), labels.to(DEVICE)
    outputs = model(data)

    vector_dim = labels.shape[-1]
    mu, var = torch.split(
        outputs, [4, 4], dim=-1
    )  # Split outputs into mu and the flattened lower triangular matrix L_flat
    var = torch.nn.functional.softplus(var) + eps

    if verbose:
        print(f"labels: {labels}")
        print(f"mu: {mu}")
        print(f"var: {var}")

    cov = torch.diag_embed(var)  # diagonal covariance matrix
    dist = MultivariateNormal(mu, cov)
    log_likelihood = dist.log_prob(labels)
    loss = -log_likelihood.mean()

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mae = F.l1_loss(mu, labels)
    r2 = r2_score(labels.cpu().detach().numpy(), mu.cpu().detach().numpy())

    if verbose:
        print(f"log_likelihood: {loss}")
        print(f"r2: {r2}")
        print(f"mae {mae}")

    return loss.item(), mae.item(), r2


def train_epoch(model, optimizer, loss_fn, dataloader, device, print_log_every=400):
    model.train()
    train_loss = 0
    train_mae = 0
    train_r2 = 0  # Added R2 score
    for batch_idx, (data, target) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Training"
    ):
        loss, mae, r2 = step(model, loss_fn, data, target, optimizer)  # Added R2 score
        train_loss += loss
        train_mae += mae
        train_r2 += r2  # Added R2 score

        if batch_idx % print_log_every == 0:
            print(f"Batch {batch_idx} Loss: {loss:.6f}  R2: {r2:.6f}    MAE: {mae:.6f}")

    return (
        train_loss / len(dataloader),
        train_mae / len(dataloader),
        train_r2 / len(dataloader),
    )  # return average R2 score


def validation_epoch(model, loss_fn, dataloader, device, print_log_every=400):
    model.eval()
    validation_loss = 0
    validation_mae = 0
    validation_r2 = 0  # Added R2 score
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Validation"
        ):
            loss, mae, r2 = step(model, loss_fn, data, target)  # Added R2 score
            validation_loss += loss
            validation_mae += mae
            validation_r2 += r2  # Added R2 score

    return (
        validation_loss / len(dataloader),
        validation_mae / len(dataloader),
        validation_r2 / len(dataloader),
    )  # return average R2 score


def train_and_test(
    args,
    model,
    word_model,
    loss_fn,
    learning_rate,
    l2_reg,
    train_dataloader,
    val_dataloader,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=l2_reg
    )
    device = torch.device(args.DEVICE)
    model.to(device)

    # Early stopping parameters
    patience = 2
    best_val_loss = float("inf")
    best_model = None
    wait = 0

    # Train model
    for epoch in range(args.EPOCHS):
        print_log("Epoch: {}".format(epoch))
        train_loss, train_mae, train_r2 = train_epoch(
            model, optimizer, loss_fn, train_dataloader, device
        )
        print_log("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))
        print_log("Epoch: {} \tTraining MAE: {:.6f}".format(epoch, train_mae))
        print_log("Epoch: {} \tTraining R2: {:.6f}".format(epoch, train_r2))

        val_loss, val_mae, val_r2 = validation_epoch(
            model, loss_fn, val_dataloader, device
        )
        print_log("Epoch: {} \tValidation Loss: {:.6f}".format(epoch, val_loss))
        print_log("Epoch: {} \tValidation MAE: {:.6f}".format(epoch, val_mae))
        print_log("Epoch: {} \tValidation R2: {:.6f}".format(epoch, val_r2))

        # Check if early stopping conditions are met
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()  # Save the model weights
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print_log(
                    "Early stopping at epoch: {}, best validation loss: {:.6f}".format(
                        epoch, best_val_loss
                    )
                )
                break

    # Use the model with the best validation loss
    model.load_state_dict(best_model_wts)

    return best_val_loss, model


def main(args):
    # Generate all combinations of hyperparameters
    HYPERPARAMS = {
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
        "l2_reg": [0.0, 0.1, 0.01, 0.0001],
        "dropout": [0.0, 0.1, 0.2, 0.5],
        "num_layers": [2, 3],
        "hidden_units": [128, 256, 512],
    }
    all_params = [
        dict(zip(HYPERPARAMS.keys(), values))
        for values in itertools.product(*HYPERPARAMS.values())
    ]

    setup_logging(args.EMB_MODEL, args.DATA_DIR)

    emb_path = args.GLOVE_PATH if args.EMB_MODEL == "glove" else args.FASTTEXT_PATH
    word_model = get_embedding_model(args.EMB_MODEL, emb_path)
    device = torch.device(args.DEVICE)

    # loss_fn = GaussianNLLLoss(full=True, eps=1e-5, reduction="mean")
    loss_fn = None

    best_params = None
    best_model = None
    best_val_loss = float("inf")

    # Load data
    train_dataset = EmbeddingDataset(args.DATA_DIR, "train", word_model)
    val_dataset = EmbeddingDataset(args.DATA_DIR, "dev", word_model)

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.BATCH_SIZE, shuffle=True
    )

    # Randomly select a hyperparameter configuration for each run
    for i in range(args.NUM_RUNS):
        print(f"\nRun {i+1} out of {args.NUM_RUNS}")
        params = random.choice(all_params)
        print_log(f"Training with parameters: {params}")

        # Create model
        model = MLP(
            args.INPUT_SIZE,
            params["hidden_units"],
            args.OUTPUT_SIZE,
            params["num_layers"],
            params["dropout"],
        )

        # Train model and get validation loss
        val_loss, model = train_and_test(
            args,
            model,
            word_model,
            loss_fn,
            params["learning_rate"],
            params["l2_reg"],
            train_dataloader,
            val_dataloader,
        )

        # Update best model and its parameters if validation loss is lower
        if val_loss < best_val_loss:
            print(f"New best model configuration with val loss {val_loss}")
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            best_params = params

    # Print the best parameters
    print_log(f"Best parameters: {best_params}")

    # Create a new model with the best parameters
    best_model = MLP(
        args.INPUT_SIZE,
        best_params["hidden_units"],
        args.OUTPUT_SIZE,
        best_params["num_layers"],
        best_params["dropout"],
    )
    # Load the state dictionary of the best model
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    # Load test data
    test_dataset = EmbeddingDataset(args.DATA_DIR, "test", word_model)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True
    )

    # Test best model
    test_loss, test_mae, test_r2 = validation_epoch(
        best_model, loss_fn, test_dataloader, device
    )
    print_log("Test Loss: {:.6f}".format(test_loss))
    print_log("Test MAE: {:.6f}".format(test_mae))
    print_log("Test R2: {:.6f}".format(test_r2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_RUNS", default=30, type=int)
    parser.add_argument("--INPUT_SIZE", default=300, type=int)
    parser.add_argument("--DEVICE", default=DEVICE, type=str)
    parser.add_argument("--OUTPUT_SIZE", default=2, type=int)
    parser.add_argument("--EPOCHS", default=50, type=int)
    parser.add_argument("--BATCH_SIZE", default=4096, type=int)
    parser.add_argument(
        "--DATA_DIR",
        default="/om/user/luwo/projects/data/baselines/baseline_data/prominence_absolute",
        type=str,
    )
    parser.add_argument(
        "--GLOVE_PATH",
        default="/om/user/luwo/projects/data/models/glove/glove.6B.300d.txt",
        type=str,
    )
    parser.add_argument(
        "--FASTTEXT_PATH",
        default="/om/user/luwo/projects/data/models/fastText/cc.en.300.bin",
        type=str,
    )
    parser.add_argument("--EMB_MODEL", default="glove", type=str)

    args = parser.parse_args()

    main(args)
