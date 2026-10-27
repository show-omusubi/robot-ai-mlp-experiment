import csv
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
    / "learning_curve_results.csv"
)

DETAIL_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "learning_curve_detail.csv"
)

ACCURACY_GRAPH = (
    Path.home()
    / "robot_ai_ws"
    / "learning_curve_accuracy.png"
)

SUCCESS_GRAPH = (
    Path.home()
    / "robot_ai_ws"
    / "learning_curve_success.png"
)


TRAIN_SIZES = [
    3,
    6,
    9,
    12,
    18,
    30,
    50,
]

# 各学習数で20回試行
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

    # LEFT = 1 -> -1
    # RIGHT = 2 -> +1
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


def count_states(df):
    """
    学習データ中に含まれている
    reflection × last_turn の状態数を数える。

    最大4種類。
    """

    states = set()

    for _, row in df.iterrows():

        state = (
            int(row["reflection"]),
            int(row["last_turn"]),
        )

        states.add(state)

    return len(states)


def count_actions(df):
    """
    学習時に経験したactionの種類数。

    最大3種類。
    """

    return df["action"].nunique()


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

        final_loss = loss.item()

    train_accuracy = calculate_accuracy(
        model,
        x_train,
        y_train,
    )

    validation_accuracy = calculate_accuracy(
        model,
        x_val,
        y_val,
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

    print()
    print(
        "Total samples:",
        len(df),
    )

    # =============================================
    # エピソード単位で学習・検証を分離
    # =============================================

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

    print(
        "Training pool:",
        len(train_pool),
    )

    print(
        "Validation:",
        len(validation_df),
    )

    print()

    # =============================================
    # 検証データの状態確認
    # =============================================

    print(
        "Validation input states:"
    )

    print(
        validation_df[
            [
                "reflection",
                "last_turn",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "reflection",
                "last_turn",
            ]
        )
    )

    # =============================================
    # 実験
    # =============================================

    summary_results = []
    detail_results = []

    for train_size in TRAIN_SIZES:

        print()
        print(
            "=============================="
        )

        print(
            "Training samples:",
            train_size,
        )

        print(
            "=============================="
        )

        validation_scores = []

        perfect_count = 0

        for repeat in range(
            REPEATS
        ):

            # -------------------------------------
            # 完全ランダム抽出
            # -------------------------------------

            sample_seed = (
                train_size * 1000
                + repeat
            )

            train_df = (
                train_pool.sample(
                    n=train_size,
                    random_state=sample_seed,
                    replace=False,
                )
            )

            state_count = count_states(
                train_df
            )

            action_count = count_actions(
                train_df
            )

            (
                train_accuracy,
                validation_accuracy,
                final_loss,
            ) = train_once(
                train_df,
                validation_df,
                sample_seed,
            )

            validation_scores.append(
                validation_accuracy
            )

            if validation_accuracy >= 0.9999:

                perfect_count += 1

            detail_results.append({
                "train_size":
                    train_size,

                "repeat":
                    repeat,

                "input_states_seen":
                    state_count,

                "actions_seen":
                    action_count,

                "train_accuracy":
                    train_accuracy,

                "validation_accuracy":
                    validation_accuracy,

                "final_loss":
                    final_loss,
            })

            print(
                f"{repeat + 1:02d}: "
                f"states={state_count}/4 "
                f"actions={action_count}/3 "
                f"train={train_accuracy:.3f} "
                f"val={validation_accuracy:.3f}"
            )

        # =========================================
        # 統計
        # =========================================

        scores = torch.tensor(
            validation_scores,
            dtype=torch.float32,
        )

        mean_accuracy = (
            scores.mean().item()
        )

        std_accuracy = (
            scores.std(
                unbiased=False
            ).item()
        )

        minimum = (
            scores.min().item()
        )

        maximum = (
            scores.max().item()
        )

        perfect_rate = (
            perfect_count
            / REPEATS
        )

        summary_results.append({
            "train_size":
                train_size,

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
            f"{perfect_count}/{REPEATS}"
        )

    # =============================================
    # CSV 保存
    # =============================================

    pd.DataFrame(
        summary_results
    ).to_csv(
        RESULT_FILE,
        index=False,
    )

    pd.DataFrame(
        detail_results
    ).to_csv(
        DETAIL_FILE,
        index=False,
    )

    # =============================================
    # Accuracy グラフ
    # =============================================

    sizes = [
        row["train_size"]
        for row in summary_results
    ]

    means = [
        row[
            "mean_validation_accuracy"
        ]
        for row in summary_results
    ]

    stds = [
        row[
            "std_validation_accuracy"
        ]
        for row in summary_results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        sizes,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Number of Training Samples"
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_GRAPH,
        dpi=150,
    )

    plt.close()

    # =============================================
    # Perfect success rate
    # =============================================

    success_rates = [
        row["perfect_rate"]
        for row in summary_results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        sizes,
        success_rates,
        marker="o",
    )

    plt.xlabel(
        "Number of Training Samples"
    )

    plt.ylabel(
        "Perfect Learning Rate"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        SUCCESS_GRAPH,
        dpi=150,
    )

    plt.close()

    # =============================================
    # 最終結果
    # =============================================

    print()
    print()
    print(
        "====================================="
    )

    print(
        "LEARNING CURVE SUMMARY"
    )

    print(
        "====================================="
    )

    for row in summary_results:

        print(
            f"N={row['train_size']:2d} "
            f"Mean={row['mean_validation_accuracy']:.4f} "
            f"Std={row['std_validation_accuracy']:.4f} "
            f"Min={row['min_validation_accuracy']:.4f} "
            f"Max={row['max_validation_accuracy']:.4f} "
            f"Perfect="
            f"{row['perfect_count']:2d}/{REPEATS}"
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
