import random
from pathlib import Path

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
    / "missing_state_balanced_results.csv"
)


REPEATS = 20
SAMPLES_PER_STATE = 10
VALIDATION_PER_STATE = 31

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


def accuracy(
    model,
    x,
    y,
):

    model.eval()

    with torch.no_grad():

        logits = model(x)

        pred = torch.argmax(
            logits,
            dim=1,
        )

        correct = (
            pred == y
        ).sum().item()

    return correct / len(y)


def build_balanced_validation(
    validation_pool,
):

    samples = []

    for index, state in enumerate(STATES):

        reflection, last_turn = state

        group = validation_pool[
            (
                validation_pool["reflection"]
                == reflection
            )
            &
            (
                validation_pool["last_turn"]
                == last_turn
            )
        ]

        if len(group) < VALIDATION_PER_STATE:

            raise RuntimeError(
                f"Not enough validation samples "
                f"for state {state}: "
                f"{len(group)} available"
            )

        sampled = group.sample(
            n=VALIDATION_PER_STATE,
            random_state=1000 + index,
            replace=False,
        )

        samples.append(sampled)

    result = pd.concat(
        samples
    )

    return result.sample(
        frac=1.0,
        random_state=999,
    )


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

        samples.append(sampled)

    result = pd.concat(samples)

    return result.sample(
        frac=1.0,
        random_state=seed,
    )


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

    return train_acc, val_acc


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

    validation_pool = df[
        df["episode"].isin(
            validation_episodes
        )
    ].copy()

    print(
        "Validation pool state counts:"
    )

    print(
        validation_pool.groupby(
            [
                "reflection",
                "last_turn",
            ]
        ).size()
    )

    validation_df = (
        build_balanced_validation(
            validation_pool
        )
    )

    print()
    print(
        "Balanced validation state counts:"
    )

    print(
        validation_df.groupby(
            [
                "reflection",
                "last_turn",
            ]
        ).size()
    )

    print()

    results = []

    for missing_state in STATES:

        label = STATE_NAME[
            missing_state
        ]

        print(
            "================================"
        )

        print(
            "Missing:",
            label,
        )

        print(
            "================================"
        )

        scores = []

        perfect_count = 0

        for repeat in range(REPEATS):

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
                train_acc,
                val_acc,
            ) = train_once(
                train_df,
                validation_df,
                seed,
            )

            scores.append(
                val_acc
            )

            if val_acc >= 0.9999:
                perfect_count += 1

            print(
                f"{repeat + 1:02d}: "
                f"train={train_acc:.3f} "
                f"val={val_acc:.3f}"
            )

        tensor_scores = torch.tensor(
            scores,
            dtype=torch.float32,
        )

        result = {
            "missing_state":
                label,

            "mean_validation_accuracy":
                tensor_scores.mean().item(),

            "std_validation_accuracy":
                tensor_scores.std(
                    unbiased=False
                ).item(),

            "min_validation_accuracy":
                tensor_scores.min().item(),

            "max_validation_accuracy":
                tensor_scores.max().item(),

            "perfect_count":
                perfect_count,

            "perfect_rate":
                perfect_count / REPEATS,
        }

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print(
        "===================================="
    )

    print(
        "BALANCED MISSING STATE SUMMARY"
    )

    print(
        "===================================="
    )

    for row in results:

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
        "CSV:",
        RESULT_FILE,
    )


if __name__ == "__main__":
    main()
