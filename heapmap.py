#!/usr/bin/env python3

"""
Generates a heatmap of R2 values of two feature and one feature loss.
It takes two csv files, one features and two feature

Usage:
    python3 heatmap.py  --two_feature /path/to/twofeature.csv  --single_feature /path/to/onefeature.csv --outdir /path/to/output_dir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

features = ["h33_gb", "h33_pm", "k27ac", "k9ac", "k4me3", "k4me1", "k36me3", "k79me2", "per1_pm", "per2_pm", "per1_gb", "per2_gb", "rnapol2"]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-timepoint CV mean R² heatmaps from feature removal CSVs.")
    parser.add_argument("--two_feature", required=True, help="Path to the two-feature removal CSV.")
    parser.add_argument("--single_feature", required=True, help="Path to the single-feature removal CSV.")
    parser.add_argument("--outdir", required=True,  help="Directory to save heatmaps.")
    return parser.parse_args()

def parse_feature(x):
    if pd.isna(x):
        return None, None
    for f1 in features:
        if x.startswith(f1 + "_"):
            f2 = x[len(f1) + 1:]
            if f2 in features:
                return f1, f2
    return None, None

def prepare_two_feature(df_two):
    plot_df = df_two[df_two["removed_feature"].notna()].copy()
    plot_df["f1"], plot_df["f2"] = zip(*plot_df["removed_feature"].apply(parse_feature))
    plot_df = plot_df.dropna(subset=["f1", "f2"])
    return plot_df

def prepare_single_feature(df_single):
    return df_single[df_single["removed_feature"].notna()].copy()

def build_matrix(tp_df, diag_tp):
    matrix = tp_df.pivot_table(
        index="f1",
        columns="f2",
        values="CV_mean_R2",
        aggfunc="mean"
    )
    for _, row in diag_tp.iterrows():
        feature = row["removed_feature"]
        if feature in features:
            matrix.loc[feature, feature] = row["CV_mean_R2"]
    return matrix

def plot_heatmap(matrix, title, ax):
    matrix = matrix.reindex(index=features, columns=features)
    mask = matrix.isna()
    sns.heatmap(
        matrix,
        ax=ax,
        mask=mask,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 6},
        linewidths=0.4,
        linecolor="lightgray",
        cbar_kws={"label": "CV mean R²", "shrink": 0.8},
        square=True,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Feature 2 (removed)", fontsize=9)
    ax.set_ylabel("Feature 1 (removed)", fontsize=9)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_two = pd.read_csv(args.two)
    df_single = pd.read_csv(args.single)

    plot_df = prepare_two_feature(df_two)
    diag_df = prepare_single_feature(df_single)

    timepoints = sorted(plot_df["timepoint"].unique())

    for tp in timepoints:
        tp_df = plot_df[plot_df["timepoint"] == tp]
        diag_tp = diag_df[diag_df["timepoint"] == tp]
        matrix = build_matrix(tp_df, diag_tp)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_heatmap(matrix, f"CV mean R² at {tp}", ax)
        plt.tight_layout()

        outpath = os.path.join(args.outdir, f"heatmap_tp{tp}.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
