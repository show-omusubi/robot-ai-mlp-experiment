import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


DATA_FILE = Path.home() / "robot_ai_ws" / "episode_training_data.csv"
RESULT_FILE = Path.home() / "robot_ai_ws" / "mlp_results.csv"
GRAPH_FILE = Path.home() / "robot_ai_ws" / "mlp_accuracy.png"
LOSS_FILE = Path.home() / "robot_ai_ws" / "mlp_loss.png"

TRAIN_SIZES = [30, 50, 100, 200, 500, 1000]

SEEDS = [0, 1, 2]

EPOCHS = 200
LEARNING_RATE = 0.01


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.network(x)


def make_features(df):
    """
    reflection:
        0 / 1

    last_turn:
        LEFT  = 1 -> -1
        RIGHT = 2 -> +1
    """

    reflection = df["reflection"].astype(float).to_numpy()

    last_turn = df["last_turn"].map({
        1: -1.0,
        2: 1.0,
    }).to_numpy()

    x = torch.tensor(
        list(zip(reflection, last_turn)),
        dtype=torch.float32,
    )

    y = torch.tensor(
        df["action"].to_numpy(),
        dtype=torch.long,
    )

    return x, y


def stratified_sample(df, n, seed):
    """
    STRAIGHT / LEFT / RIGHT が極端に偏らないように
    各クラスから均等に抽出する。
    """

    random.seed(seed)

    samples = []

    base = n // 3
    remainder = n % 3

    for action in [0, 1, 2]:
        count = base

        if action < remainder:
            count += 1

        group = df[df["action"] == action]

        sampled = group.sample(
            n=count,
            random_state=seed + action,
            replace=False,
        )

        samples.append(sampled)

    result = pd.concat(samples)

    result = result.sample(
        frac=1.0,
        random_state=seed,
    )

    return result


def accuracy(model, x, y):
    model.eval()

    with torch.no_grad():
        logits = model(x)

        prediction = torch.argmax(
            logits,
            dim=1,
        )

        correct = (
            prediction == y
        ).sum().item()

    return correct / len(y)


def train_once(
    train_df,
    validation_df,
    seed,
):
    random.seed(seed)
    torch.manual_seed(seed)

    x_train, y_train = make_features(train_df)
    x_val, y_val = make_features(validation_df)

    model = MLP()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_losses = []
    val_losses = []

    for _ in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        logits = model(x_train)

        loss = criterion(
            logits,
            y_train,
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

        model.eval()

        with torch.no_grad():

            val_logits = model(x_val)

            val_loss = criterion(
                val_logits,
                y_val,
            )

            val_losses.append(
                val_loss.item()
            )

    train_acc = accuracy(
        model,
        x_train,
        y_train,
    )

    val_acc = accuracy(
        model,
        x_val,
        y_val,
    )

    return (
        train_acc,
        val_acc,
        train_losses,
        val_losses,
    )


def main():

    print("Loading:")
    print(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    print()
    print("Total samples:", len(df))
    print()
    print("Action distribution:")
    print(df["action"].value_counts().sort_index())

    # ---------------------------------------------
    # エピソード単位で学習用と検証用に分離
    #
    # 0-47 : training pool
    # 48-59: validation
    # ---------------------------------------------

    episodes = sorted(
        df["episode"].unique()
    )

    split_index = int(
        len(episodes) * 0.8
    )

    train_episodes = episodes[
        :split_index
    ]

    validation_episodes = episodes[
        split_index:
    ]

    train_pool = df[
        df["episode"].isin(
            train_episodes
        )
    ]

    validation_df = df[
        df["episode"].isin(
            validation_episodes
        )
    ]

    print()
    print(
        "Training pool:",
        len(train_pool),
    )

    print(
        "Validation:",
        len(validation_df),
    )

    results = []

    loss_history = {}

    for train_size in TRAIN_SIZES:

        print()
        print(
            "Training samples:",
            train_size,
        )

        seed_train_acc = []
        seed_val_acc = []

        for seed in SEEDS:

            train_df = stratified_sample(
                train_pool,
                train_size,
                seed,
            )

            (
                train_acc,
                val_acc,
                train_losses,
                val_losses,
            ) = train_once(
                train_df,
                validation_df,
                seed,
            )

            seed_train_acc.append(
                train_acc
            )

            seed_val_acc.append(
                val_acc
            )

            print(
                f"  seed={seed} "
                f"train={train_acc:.4f} "
                f"validation={val_acc:.4f}"
            )

            if seed == 0:
                loss_history[
                    train_size
                ] = (
                    train_losses,
                    val_losses,
                )

        mean_train = sum(
            seed_train_acc
        ) / len(seed_train_acc)

        mean_val = sum(
            seed_val_acc
        ) / len(seed_val_acc)

        results.append({
            "train_size": train_size,
            "train_accuracy": mean_train,
            "validation_accuracy": mean_val,
        })

    # =============================================
    # CSV保存
    # =============================================

    with open(
        RESULT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "train_size",
                "train_accuracy",
                "validation_accuracy",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =============================================
    # Accuracy graph
    # =============================================

    sizes = [
        row["train_size"]
        for row in results
    ]

    train_acc = [
        row["train_accuracy"]
        for row in results
    ]

    val_acc = [
        row["validation_accuracy"]
        for row in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        sizes,
        train_acc,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        sizes,
        val_acc,
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel(
        "Number of Training Samples"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=150,
    )

    plt.close()

    # =============================================
    # Loss graph
    # =============================================

    plt.figure(
        figsize=(9, 6)
    )

    for size in TRAIN_SIZES:

        train_losses, _ = (
            loss_history[size]
        )

        plt.plot(
            train_losses,
            label=f"N={size}",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        LOSS_FILE,
        dpi=150,
    )

    plt.close()

    # =============================================
    # 結果表示
    # =============================================

    print()
    print("==============================")
    print("RESULT")
    print("==============================")

    for row in results:

        print(
            f"N={row['train_size']:4d} "
            f"Train={row['train_accuracy']:.4f} "
            f"Validation={row['validation_accuracy']:.4f}"
        )

    print()
    print(
        "CSV:",
        RESULT_FILE,
    )

    print(
        "Accuracy graph:",
        GRAPH_FILE,
    )

    print(
        "Loss graph:",
        LOSS_FILE,
    )


if __name__ == "__main__":
    main()
