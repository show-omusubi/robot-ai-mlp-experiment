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
    / "noise_robustness_results.csv"
)

GRAPH_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "noise_robustness_accuracy.png"
)


STATES = [
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
]

TRAIN_PER_STATE = 50
VALIDATION_PER_STATE = 31

REPEATS = 20
EPOCHS = 200
LEARNING_RATE = 0.01

# 学習時に与えるノイズ
TRAIN_NOISE_LEVELS = [
    0.00,
    0.15,
]

# 評価時に与えるノイズ
TEST_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


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


def make_clean_features(df):

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


def add_noise(
    x,
    sigma,
):

    noisy = x.clone()

    if sigma > 0.0:

        noise = (
            torch.randn(
                len(noisy)
            )
            * sigma
        )

        noisy[:, 0] = (
            noisy[:, 0]
            + noise
        )

        # reflectionは0～1に制限
        noisy[:, 0] = torch.clamp(
            noisy[:, 0],
            0.0,
            1.0,
        )

    return noisy


def balanced_sample(
    df,
    number_per_state,
    seed,
):

    samples = []

    for index, (
        reflection,
        last_turn,
    ) in enumerate(STATES):

        group = df[
            (
                df["reflection"]
                == reflection
            )
            &
            (
                df["last_turn"]
                == last_turn
            )
        ]

        if len(group) < number_per_state:

            raise RuntimeError(
                f"Not enough samples "
                f"for state "
                f"({reflection}, "
                f"{last_turn}). "
                f"Available={len(group)}"
            )

        sampled = group.sample(
            n=number_per_state,
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

    return result.sample(
        frac=1.0,
        random_state=seed,
    )


def train_model(
    train_df,
    train_noise,
    seed,
):

    torch.manual_seed(
        seed
    )

    x_clean, y_train = (
        make_clean_features(
            train_df
        )
    )

    model = MLP()

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
    )

    for _ in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()

        # 毎epoch違うノイズを加える
        x_train = add_noise(
            x_clean,
            train_noise,
        )

        logits = model(
            x_train
        )

        loss = criterion(
            logits,
            y_train,
        )

        loss.backward()

        optimizer.step()

    return model


def evaluate(
    model,
    validation_df,
    test_noise,
    seed,
):

    # 評価ノイズの再現性
    torch.manual_seed(
        seed
    )

    x_clean, y = (
        make_clean_features(
            validation_df
        )
    )

    x_test = add_noise(
        x_clean,
        test_noise,
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            x_test
        )

        prediction = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        correct = (
            prediction == y
        ).sum().item()

    return (
        correct
        / len(y)
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

    validation_pool = df[
        df["episode"].isin(
            validation_episodes
        )
    ].copy()

    print(
        "Training state counts:"
    )

    print(
        train_pool.groupby(
            [
                "reflection",
                "last_turn",
            ]
        ).size()
    )

    print()

    print(
        "Validation state counts:"
    )

    print(
        validation_pool.groupby(
            [
                "reflection",
                "last_turn",
            ]
        ).size()
    )

    # Validationは常に同じ
    # 各状態31件の完全均等
    validation_df = (
        balanced_sample(
            validation_pool,
            VALIDATION_PER_STATE,
            999,
        )
    )

    results = []

    for train_noise in (
        TRAIN_NOISE_LEVELS
    ):

        print()
        print(
            "=================================="
        )

        print(
            f"TRAIN NOISE = "
            f"{train_noise:.2f}"
        )

        print(
            "=================================="
        )

        for test_noise in (
            TEST_NOISE_LEVELS
        ):

            scores = []

            for repeat in range(
                REPEATS
            ):

                seed = (
                    int(
                        train_noise
                        * 10000
                    )
                    + int(
                        test_noise
                        * 1000
                    )
                    + repeat
                )

                # 毎回学習データも
                # 少し変える
                train_df = (
                    balanced_sample(
                        train_pool,
                        TRAIN_PER_STATE,
                        seed,
                    )
                )

                model = (
                    train_model(
                        train_df,
                        train_noise,
                        seed,
                    )
                )

                accuracy = (
                    evaluate(
                        model,
                        validation_df,
                        test_noise,
                        seed + 50000,
                    )
                )

                scores.append(
                    accuracy
                )

            score_tensor = (
                torch.tensor(
                    scores,
                    dtype=torch.float32,
                )
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

            results.append({
                "train_noise":
                    train_noise,

                "test_noise":
                    test_noise,

                "mean_accuracy":
                    mean_accuracy,

                "std_accuracy":
                    std_accuracy,

                "min_accuracy":
                    minimum,

                "max_accuracy":
                    maximum,
            })

            print(
                f"Test noise="
                f"{test_noise:.2f} "
                f"Mean="
                f"{mean_accuracy:.4f} "
                f"Std="
                f"{std_accuracy:.4f} "
                f"Min="
                f"{minimum:.4f} "
                f"Max="
                f"{maximum:.4f}"
            )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    # =========================================
    # グラフ
    # =========================================

    plt.figure(
        figsize=(8, 5)
    )

    for train_noise in (
        TRAIN_NOISE_LEVELS
    ):

        part = result_df[
            result_df["train_noise"]
            == train_noise
        ]

        plt.errorbar(
            part["test_noise"],
            part["mean_accuracy"],
            yerr=part["std_accuracy"],
            marker="o",
            capsize=4,
            label=(
                f"Train noise="
                f"{train_noise:.2f}"
            ),
        )

    plt.xlabel(
        "Sensor Noise (sigma)"
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

    plt.legend()

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
        "NOISE ROBUSTNESS SUMMARY"
    )

    print(
        "===================================="
    )

    for train_noise in (
        TRAIN_NOISE_LEVELS
    ):

        print()
        print(
            f"Train noise = "
            f"{train_noise:.2f}"
        )

        part = result_df[
            result_df["train_noise"]
            == train_noise
        ]

        for _, row in (
            part.iterrows()
        ):

            print(
                f"  Test="
                f"{row['test_noise']:.2f} "
                f"Mean="
                f"{row['mean_accuracy']:.4f} "
                f"Std="
                f"{row['std_accuracy']:.4f}"
            )

    print()
    print(
        "CSV:",
        RESULT_FILE,
    )

    print(
        "Graph:",
        GRAPH_FILE,
    )


if __name__ == "__main__":
    main()
