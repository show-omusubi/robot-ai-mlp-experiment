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
    / "repetition_effect_results.csv"
)

DETAIL_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "repetition_effect_detail.csv"
)

ACCURACY_GRAPH = (
    Path.home()
    / "robot_ai_ws"
    / "repetition_effect_accuracy.png"
)

SUCCESS_GRAPH = (
    Path.home()
    / "robot_ai_ws"
    / "repetition_effect_success.png"
)


REPETITIONS_PER_STATE = [
    1,
    2,
    3,
    5,
    10,
    20,
]

REPEATS = 20

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


def sample_all_states(
    train_pool,
    repetitions,
    seed,
):
    """
    4種類の入力状態を必ず含める。

    各状態からrepetitions件ずつ抽出する。
    """

    samples = []

    states = [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]

    for index, (
        reflection,
        last_turn,
    ) in enumerate(states):

        group = train_pool[
            (
                train_pool[
                    "reflection"
                ] == reflection
            )
            &
            (
                train_pool[
                    "last_turn"
                ] == last_turn
            )
        ]

        sampled = group.sample(
            n=repetitions,
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

    final_loss = None

    for _ in range(
        EPOCHS
    ):

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

        final_loss = (
            loss.item()
        )

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
        final_loss,
    )


def main():

    print(
        "Loading:",
        DATA_FILE,
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Total samples:",
        len(df),
    )

    # =========================================
    # Episode単位で学習・検証を分離
    # =========================================

    episodes = sorted(
        df["episode"].unique()
    )

    split_index = int(
        len(episodes) * 0.8
    )

    train_episodes = (
        episodes[
            :split_index
        ]
    )

    validation_episodes = (
        episodes[
            split_index:
        ]
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

    print(
        "Training pool:",
        len(train_pool),
    )

    print(
        "Validation:",
        len(validation_df),
    )

    # =========================================
    # 各状態の件数確認
    # =========================================

    print()
    print(
        "Available states:"
    )

    state_counts = (
        train_pool
        .groupby(
            [
                "reflection",
                "last_turn",
            ]
        )
        .size()
    )

    print(
        state_counts
    )

    summary_results = []
    detail_results = []

    # =========================================
    # 実験
    # =========================================

    for repetitions in (
        REPETITIONS_PER_STATE
    ):

        total_samples = (
            repetitions * 4
        )

        print()
        print(
            "================================"
        )

        print(
            f"Repetitions/state: "
            f"{repetitions}"
        )

        print(
            f"Total samples: "
            f"{total_samples}"
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
                repetitions
                * 1000
                + repeat
            )

            train_df = (
                sample_all_states(
                    train_pool,
                    repetitions,
                    seed,
                )
            )

            (
                train_accuracy,
                validation_accuracy,
                final_loss,
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
                "repetitions_per_state":
                    repetitions,

                "total_samples":
                    total_samples,

                "repeat":
                    repeat,

                "train_accuracy":
                    train_accuracy,

                "validation_accuracy":
                    validation_accuracy,

                "final_loss":
                    final_loss,
            })

            print(
                f"{repeat + 1:02d}: "
                f"train="
                f"{train_accuracy:.3f} "
                f"val="
                f"{validation_accuracy:.3f}"
            )

        tensor_scores = (
            torch.tensor(
                scores,
                dtype=torch.float32,
            )
        )

        mean_accuracy = (
            tensor_scores
            .mean()
            .item()
        )

        std_accuracy = (
            tensor_scores
            .std(
                unbiased=False
            )
            .item()
        )

        minimum = (
            tensor_scores
            .min()
            .item()
        )

        maximum = (
            tensor_scores
            .max()
            .item()
        )

        perfect_rate = (
            perfect_count
            / REPEATS
        )

        summary_results.append({
            "repetitions_per_state":
                repetitions,

            "total_samples":
                total_samples,

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

        print()
        print(
            f"Mean    = "
            f"{mean_accuracy:.4f}"
        )

        print(
            f"Std     = "
            f"{std_accuracy:.4f}"
        )

        print(
            f"Min     = "
            f"{minimum:.4f}"
        )

        print(
            f"Max     = "
            f"{maximum:.4f}"
        )

        print(
            f"Perfect = "
            f"{perfect_count}/"
            f"{REPEATS}"
        )

    # =========================================
    # CSV
    # =========================================

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

    # =========================================
    # Accuracy graph
    # =========================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        summary_df[
            "total_samples"
        ],
        summary_df[
            "mean_validation_accuracy"
        ],
        yerr=summary_df[
            "std_validation_accuracy"
        ],
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Total Training Samples"
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        ACCURACY_GRAPH,
        dpi=150,
    )

    plt.close()

    # =========================================
    # Perfect success graph
    # =========================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary_df[
            "total_samples"
        ],
        summary_df[
            "perfect_rate"
        ],
        marker="o",
    )

    plt.xlabel(
        "Total Training Samples"
    )

    plt.ylabel(
        "Perfect Learning Rate"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        SUCCESS_GRAPH,
        dpi=150,
    )

    plt.close()

    # =========================================
    # Summary
    # =========================================

    print()
    print()
    print(
        "===================================="
    )

    print(
        "REPETITION EFFECT SUMMARY"
    )

    print(
        "===================================="
    )

    for row in summary_results:

        print(
            f"R={row['repetitions_per_state']:2d} "
            f"N={row['total_samples']:3d} "
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
        "Accuracy graph:",
        ACCURACY_GRAPH,
    )

    print(
        "Success graph:",
        SUCCESS_GRAPH,
    )


if __name__ == "__main__":
    main()
