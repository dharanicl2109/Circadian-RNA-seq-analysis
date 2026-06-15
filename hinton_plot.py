import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-timepoint Hinton plots from feature removal CSVs.")
    parser.add_argument("--two", required=True, metavar="TWO_FEATURE_CSV", help="Path to the two-feature removal CSV.")
    parser.add_argument("--single", required=True, metavar="SINGLE_FEATURE_CSV", help="Path to the single-feature removal CSV.")
    parser.add_argument("--outdir", default=".", metavar="OUTPUT_DIR", help="Directory to save Hinton plot PNGs (default: current directory).")
    return parser.parse_args()

features = ["h33_gb", "h33_pm", "k27ac", "k9ac", "k4me3", "k4me1", "k36me3", "k79me2", "per1_pm", "per2_pm", "per1_gb", "per2_gb", "rnapol2"]

def parse_feature(x):
    if pd.isna(x):
        return None, None
    for f1 in features:
        if x.startswith(f1 + "_"):
            f2 = x[len(f1) + 1:]
            if f2 in features:
                return f1, f2
            return f1, None
    return None, None

def prepare_two_feature(df_two):
    baseline = (
        df_two[df_two["removed_feature"].isna()]
        .set_index("timepoint")["CV_mean_R2"]
    )
    df_two["baseline_R2"] = df_two["timepoint"].map(baseline)
    df_two["R2_difference"] = df_two["baseline_R2"] - df_two["CV_mean_R2"]
    plot_df = df_two[df_two["removed_feature"].notna()].copy()
    plot_df["f1"], plot_df["f2"] = zip(*plot_df["removed_feature"].apply(parse_feature))
    plot_df = plot_df.dropna(subset=["f1", "f2"])
    return plot_df

def prepare_single_feature(df_single):
    baseline_single = (
        df_single[df_single["removed_feature"].isna()]
        .set_index("timepoint")["CV_mean_R2"]
    )
    df_single["baseline_R2"] = df_single["timepoint"].map(baseline_single)
    df_single["R2_difference"] = df_single["baseline_R2"] - df_single["CV_mean_R2"]
    return df_single[df_single["removed_feature"].notna()].copy()

def build_matrix(tp_df, diag_tp):
    matrix = tp_df.pivot_table(
        index="f1",
        columns="f2",
        values="R2_difference",
        aggfunc="mean"
    )
    for _, row in diag_tp.iterrows():
        feature = row["removed_feature"]
        value = row["R2_difference"]
        if feature in features:
            matrix.loc[feature, feature] = value
    return matrix

def plot_matrix(matrix, title, outpath):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("gray")
    matrix = matrix.reindex(index=features, columns=features).fillna(0)
    max_weight = np.max(np.abs(matrix.values))
    if max_weight == 0:
        max_weight = 1
    for (x, y), w in np.ndenumerate(matrix.values):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle(
            (y - size / 2, x - size / 2),
            size,
            size,
            facecolor=color,
            edgecolor=color,
            linewidth=0
        )
        ax.add_patch(rect)
    ax.set_xlim(-0.5, len(features) - 0.5)
    ax.set_ylim(-0.5, len(features) - 0.5)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")

def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_two = pd.read_csv(args.two)
    df_single = pd.read_csv(args.single)

    plot_df = prepare_two_feature(df_two)
    diag_df = prepare_single_feature(df_single)

    timepoints = sorted(plot_df["timepoint"].unique())

    for tp in timepoints:
        print(f"Plotting timepoint: {tp}")
        tp_df = plot_df[plot_df["timepoint"] == tp]
        diag_tp = diag_df[diag_df["timepoint"] == tp]
        matrix = build_matrix(tp_df, diag_tp)
        outpath = os.path.join(args.outdir, f"hinton_tp{tp}.png")
        plot_matrix(matrix, f"Timepoint {tp}", outpath)

if __name__ == "__main__":
    main()
