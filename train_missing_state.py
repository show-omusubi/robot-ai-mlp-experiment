import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


DATA_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "episode_training_data.csv"
)

RESULT_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "missing_state_results.csv"
)

DETAIL_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "missing_state_detail.csv"
)

GRAPH_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "missing_state_accuracy.png"
)


REPEATS = 20
SAMPLES_PER_STATE = 10

EPOCHS = 200
LEARNING_RATE = 0.01


STATES = [
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
]


STATE_NAME = {
    (0, 1): "R0_LEFT",
    (0, 2): "R0_RIGHT",
    (1, 1): "R1_LEFT",
    (1, 2): "R1_RIGHT",
}


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

    reflection = (
        df["reflection"]
        .astype(float)
        .to_numpy()
    )

    last_turn = (
        df["last_turn"]
        .map({
            1: -1.0,
            2: 1.0,
        })
        .to_numpy()
    )

    x = torch.tensor(
        list(
            zip(
                reflection,
                last_turn,
            )
        ),
        dtype=torch.float32,
    )

    y = torch.tensor(
        df["action"].to_numpy(),
        dtype=torch.long,
    )

    return x, y


def calculate_accuracy(
    model,
    x,
    y,
):

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


def sample_without_state(
    train_pool,
    missing_state,
    seed,
):

    samples = []

    for index, state in enumerate(STATES):

        if state == missing_state:
            continue

        reflection, last_turn = state

        group = train_pool[
            (
                train_pool["reflection"]
                == reflection
            )
            &
            (
                train_pool["last_turn"]
                == last_turn
            )
        ]

        sampled = group.sample(
            n=SAMPLES_PER_STATE,
            random_state=(
                seed
                + index * 100
            ),
            replace=False,
        )

        samples.append(
            sampled
        )

    result = pd.concat(
        samples
    )

    result = result.sample(
        frac=1.0,
        random_state=seed,
    )

    return result


def train_once(
    train_df,
    validation_df,
    seed,
):

    random.seed(seed)
    torch.manual_seed(seed)

    x_train, y_train = make_features(
        train_df
    )

    x_val, y_val = make_features(
        validation_df
    )

    model = MLP()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for _ in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        logits = model(
            x_train
        )

        loss = criterion(
            logits,
            y_train,
        )

        loss.backward()

        optimizer.step()

    train_accuracy = (
        calculate_accuracy(
            model,
            x_train,
            y_train,
        )
    )

    validation_accuracy = (
        calculate_accuracy(
            model,
            x_val,
            y_val,
        )
    )

    return (
        train_accuracy,
        validation_accuracy,
    )


def main():

    df = pd.read_csv(
        DATA_FILE
    )

    episodes = sorted(
        df["episode"].unique()
    )

    split_index = int(
        len(episodes) * 0.8
    )

    train_episodes = (
        episodes[:split_index]
    )

    validation_episodes = (
        episodes[split_index:]
    )

    train_pool = df[
        df["episode"].isin(
            train_episodes
        )
    ].copy()

    validation_df = df[
        df["episode"].isin(
            validation_episodes
        )
    ].copy()

    summary_results = []
    detail_results = []

    for missing_state in STATES:

        label = STATE_NAME[
            missing_state
        ]

        print()
        print(
            "================================"
        )

        print(
            "Missing state:",
            label,
        )

        print(
            "================================"
        )

        scores = []

        perfect_count = 0

        for repeat in range(
            REPEATS
        ):

            seed = (
                STATES.index(
                    missing_state
                )
                * 1000
                + repeat
            )

            train_df = (
                sample_without_state(
                    train_pool,
                    missing_state,
                    seed,
                )
            )

            (
                train_accuracy,
                validation_accuracy,
            ) = train_once(
                train_df,
                validation_df,
                seed,
            )

            scores.append(
                validation_accuracy
            )

            if (
                validation_accuracy
                >= 0.9999
            ):
                perfect_count += 1

            detail_results.append({
                "missing_state":
                    label,

                "repeat":
                    repeat,

                "train_accuracy":
                    train_accuracy,

                "validation_accuracy":
                    validation_accuracy,
            })

            print(
                f"{repeat + 1:02d}: "
                f"train="
                f"{train_accuracy:.3f} "
                f"val="
                f"{validation_accuracy:.3f}"
            )

        score_tensor = torch.tensor(
            scores,
            dtype=torch.float32,
        )

        mean_accuracy = (
            score_tensor
            .mean()
            .item()
        )

        std_accuracy = (
            score_tensor
            .std(
                unbiased=False
            )
            .item()
        )

        minimum = (
            score_tensor
            .min()
            .item()
        )

        maximum = (
            score_tensor
            .max()
            .item()
        )

        perfect_rate = (
            perfect_count
            / REPEATS
        )

        summary_results.append({
            "missing_state":
                label,

            "mean_validation_accuracy":
                mean_accuracy,

            "std_validation_accuracy":
                std_accuracy,

            "min_validation_accuracy":
                minimum,

            "max_validation_accuracy":
                maximum,

            "perfect_count":
                perfect_count,

            "perfect_rate":
                perfect_rate,
        })

    summary_df = pd.DataFrame(
        summary_results
    )

    detail_df = pd.DataFrame(
        detail_results
    )

    summary_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    detail_df.to_csv(
        DETAIL_FILE,
        index=False,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        summary_df[
            "missing_state"
        ],
        summary_df[
            "mean_validation_accuracy"
        ],
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=150,
    )

    plt.close()

    print()
    print(
        "===================================="
    )

    print(
        "MISSING STATE SUMMARY"
    )

    print(
        "===================================="
    )

    for row in summary_results:

        print(
            f"{row['missing_state']:10s} "
            f"Mean="
            f"{row['mean_validation_accuracy']:.4f} "
            f"Std="
            f"{row['std_validation_accuracy']:.4f} "
            f"Min="
            f"{row['min_validation_accuracy']:.4f} "
            f"Max="
            f"{row['max_validation_accuracy']:.4f} "
            f"Perfect="
            f"{row['perfect_count']:2d}/"
            f"{REPEATS}"
        )

    print()
    print(
        "Summary CSV:",
        RESULT_FILE,
    )

    print(
        "Detail CSV:",
        DETAIL_FILE,
    )

    print(
        "Graph:",
        GRAPH_FILE,
    )


if __name__ == "__main__":
    main()
