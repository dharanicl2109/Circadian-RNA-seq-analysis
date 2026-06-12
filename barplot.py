#!/usr/bin/env python3

"""
Generates a barplot using R2 values of features at different timepoints.
The first row consists of timepoints and the first column coonsists of PTMs.

Usage:
    python3 barplot.py ---input_r2 path/to/input_r2.cs
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate barplot of R2 values of each PTM timepoint-wise ")
    parser.add_argument("--input_r2", required=True, help="Path to CSV file containing R2 values")
    args = parser.parse_args()

    df = pd.read_csv(args.input_r2)

    # make timepoint the index
    df = df.set_index("timepoint")

    # transpose so PTMs become x-axis groups
    df_t = df.T

    # plot grouped bars
    ax = df_t.plot(kind="bar", figsize=(10,6))

    plt.xlabel("PTM")
    plt.ylabel("R2")
    plt.title("R2 values of PTMs across timepoints")
    plt.legend(title="Timepoint")

    plt.tight_layout()
    plt.show()
