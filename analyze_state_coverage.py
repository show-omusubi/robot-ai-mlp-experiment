from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DETAIL_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "learning_curve_detail.csv"
)

SUMMARY_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "state_coverage_summary.csv"
)

GRAPH_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "state_coverage_accuracy.png"
)

SIZE_GRAPH_FILE = (
    Path.home()
    / "robot_ai_ws"
    / "state_coverage_by_train_size.png"
)


def main():

    df = pd.read_csv(
        DETAIL_FILE
    )

    print("Total experiments:", len(df))
    print()

    # ============================================
    # 状態数ごとの集計
    # ============================================

    summary = (
        df.groupby("input_states_seen")
        ["validation_accuracy"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max",
        ])
        .reset_index()
    )

    print(
        "=================================="
    )
    print(
        "STATE COVERAGE SUMMARY"
    )
    print(
        "=================================="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ============================================
    # 完全成功率
    # ============================================

    perfect = (
        df.assign(
            perfect=(
                df["validation_accuracy"]
                >= 0.9999
            )
        )
        .groupby(
            "input_states_seen"
        )["perfect"]
        .agg([
            "count",
            "sum",
            "mean",
        ])
        .reset_index()
    )

    print()
    print(
        "=================================="
    )
    print(
        "PERFECT LEARNING RATE"
    )
    print(
        "=================================="
    )

    print(
        perfect.to_string(
            index=False
        )
    )

    # ============================================
    # 状態数 vs Accuracy
    # ============================================

    states = (
        summary[
            "input_states_seen"
        ]
        .to_numpy()
    )

    means = (
        summary[
            "mean"
        ]
        .to_numpy()
    )

    stds = (
        summary[
            "std"
        ]
        .fillna(0)
        .to_numpy()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        states,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Number of Input States Seen"
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.xticks(
        sorted(states)
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=150,
    )

    plt.close()

    # ============================================
    # train_sizeごとに状態網羅率を確認
    # ============================================

    coverage = (
        df.groupby(
            [
                "train_size",
                "input_states_seen",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print()
    print(
        "=================================="
    )
    print(
        "STATE COVERAGE BY TRAIN SIZE"
    )
    print(
        "=================================="
    )

    print(
        coverage.to_string()
    )

    coverage.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
    )

    plt.xlabel(
        "Number of Training Samples"
    )

    plt.ylabel(
        "Number of Runs"
    )

    plt.legend(
        title="Input States Seen"
    )

    plt.tight_layout()

    plt.savefig(
        SIZE_GRAPH_FILE,
        dpi=150,
    )

    plt.close()

    # ============================================
    # 追加: action種類数との関係
    # ============================================

    action_summary = (
        df.groupby(
            "actions_seen"
        )["validation_accuracy"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max",
        ])
        .reset_index()
    )

    print()
    print(
        "=================================="
    )
    print(
        "ACTION COVERAGE SUMMARY"
    )
    print(
        "=================================="
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary CSV:",
        SUMMARY_FILE,
    )

    print(
        "State accuracy graph:",
        GRAPH_FILE,
    )

    print(
        "Coverage graph:",
        SIZE_GRAPH_FILE,
    )


if __name__ == "__main__":
    main()
