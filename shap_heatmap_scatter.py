#!/usr/bin/env python3

"""
Generates a CSV file of SHAP values for all genes in the test set.
It takes the CLOCK-labelled dataset and the output directory as inputs.

Usage:
    python3 shap_analysis.py --input path/to/clock_labelled_dataset.csv --output-dir /path/to/output_dir
"""
    

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest, compute SHAP values, and save to CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to save shap_value_all_genes.csv")
    return parser.parse_args()
    
def main():
    args = parse_args()
    os.mkdirs(args.output_dir, exist_ok=True)
    
    # features
    features = [
        "h3.3_gene_body_ct0",
        "h3.3_promoters_ct0",
        "h3k27ac_ct0",
        "h3k9ac_ct0",
        "h3k4me3_ct0",
        "h3k4me1_ct0",
        "h3k36me3_gene_body_ct0",
        "h3k79me2_gene_body_ct0",
        "per1_promoters_ct0",
        'per1_gene_body_ct0',
        "per2_promoters_ct0",
        'per2_gene_body_ct0',
        "rnapol2_promoter_ct0"
    ]
    
    # class label column
    class_col = 'label'
    
    # -----data loading-----
    with tqdm(total=1, desc="Reading CSV", unit="file") as pbar:
    data = pd.read_csv(args.input, na_values=["NA", "null", "?", " "], engine="python")
    pbar.update(1)
    print(data.columns)
    
    # gene names
    gene_names = data.iloc[:, 0].values
    
    # input
    X = data[features]

    # RNA-seq expression values
    y = data["ct0_rpkm_cm_avg"]
    
    # ------train test split------
    with tqdm(total=1, desc="Train-test split", unit="step") as pbar:
        X_train, X_test, y_train, y_test, genes_train, genes_test = train_test_split(X, y, gene_names, test_size=0.30, random_state=89)
        pbar.update(1)
        
    # ------scaling-----
    with tqdm(total=2, desc="Scaling features", unit="step") as pbar:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        pbar.update(1)
        X_test_scaled = scaler.transform(X_test)
        pbar.update(1)
        
    # log transform RNA expression
    with tqdm(total=2, desc="Log transforming target", unit="step") as pbar:
        y_train_scaled = np.log1p(y_train)
        pbar.update(1)
        y_test_scaled = np.log1p(y_test)
        pbar.update(1)
        
    # ------model training-----
    with tqdm(total=1, desc="Training Random Forest", unit="step") as pbar:
        model = RandomForestRegressor(n_estimators=200, min_samples_split=2, max_depth=None, random_state=999, n_jobs=-1)
        model.fit(X_train_scaled, y_train_scaled)
        pbar.update(1)
        
    # -----prediction and evaluation-----
    with tqdm(total=1, desc="Predicting", unit="step") as pbar:
        y_pred = model.predict(X_test_scaled)
        pbar.update(1)

    r2 = r2_score(y_test_scaled, y_pred)
    mae = mean_absolute_error(y_test_scaled, y_pred)

    print("R2:", r2)
    print("MAE:", mae)
    
    

    # -----SHAP analysis-----

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

    with tqdm(total=1, desc="Initializing SHAP explainer", unit="step") as pbar:
        explainer = shap.TreeExplainer(model)
        pbar.update(1)

    # compute SHAP values for all test genes
    with tqdm(total=1, desc="Computing SHAP values for all test genes", unit="step") as pbar:
        shap_values_all = explainer(X_test_scaled_df, check_additivity=False)
        pbar.update(1)
        
    # -----SHAP values dataframe------
    shap_csv_path = os.path.join(args.output_dir, "shap_value_all_test_genes.csv")
    with tqdm(total=1, desc="Building SHAP dataframe", unit="step") as pbar:
        shap_df = pd.DataFrame(shap_values_all.values, columns=features)
        shap_df.insert(0, 'gene', genes_test)
        shap_df.insert(1, 'class', data.loc[X_test.index, class_col].values)
        shap_df.to_csv(shap_csv_path, index=False)
        pbar.update(1)
    
    print(f"SHAP values saved to {output_dir}/shap_value_all_test_genes.csv")
    
    
    # -----SHAP heatmap-----
    X_test_df = X_test.copy()
    X_test_df['__gene__'] = genes_test
    X_test_df['__class__'] = data.loc[X_test.index, class_col].values
    
     for class_label in tqdm([0, 1], desc="Generating heatmaps", unit="class"):
        class_mask = X_test_df['__class__'] == class_label
        X_test_class = X_test_df[class_mask]
        print(f"\nTotal class-{class_label} rows in test set: {len(X_test_class)}")
 
        if len(X_test_class) == 0:
            print(f"No class-{class_label} in the test set.")
            continue
 
        # randomly sample upto 3000 rows
        sample_size = min(3000, len(X_test_class))
        X_test_class_sample = X_test_class.sample(n=sample_size, random_state=89)
 
        # scale the sampled features
        X_heatmap_scaled = scaler.transform(X_test_class_sample[features])
        X_heatmap_scaled_df = pd.DataFrame(X_heatmap_scaled, columns=features)
 
        # compute SHAP values
        shap_values_heatmap = explainer(X_heatmap_scaled_df, check_additivity=False)
 
        # plot and save heatmap
        plt.figure()
        shap.plots.heatmap(shap_values_heatmap, show=False, max_display=len(features))
        plt.title(f"SHAP Heatmap - Class {class_label} Genes (n={sample_size})")
        plt.savefig(f"{output_dir}/shap_heatmap_class{class_label}_n_{sample_size}_confirmation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
    # -----SHAP scatter plot-----
    def save_scatter(feature, shap_values_class, class_label, class_scatter_dir):
        plt.figure()
        shap.plots.scatter(shap_values_class[:, feature], show=False)
        plt.title(f"SHAP Scatter - {feature} (Class {class_label})")
        plt.savefig(f"{class_scatter_dir}/shap_scatter_{feature}_class{class_label}_confirmation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    for class_label in tqdm([0, 1], desc="Scatter plot classes", unit="class"):
        class_mask = X_test_df['__class__'] == class_label
        class_indices = X_test_df[class_mask].index
 
        class_indices_set = set(class_indices)
        positions = [i for i, idx in enumerate(X_test.index) if idx in class_indices_set]
 
        shap_values_class = shap_values_all[positions]
 
        class_scatter_dir = os.path.join(output_dir, f"class_{class_label}_confirmation")
        os.makedirs(class_scatter_dir, exist_ok=True)
 
        with parallel_backend('threading'):
            Parallel(n_jobs=-1)(
                delayed(save_scatter)(feature, shap_values_class, class_label, class_scatter_dir)
                for feature in tqdm(features, desc=f"  Scatter plots class {class_label}", unit="feature", leave=False)
            )
        print(f"Class-{class_label} scatter plots saved to {class_scatter_dir}")
    
 
 
 if __name__ == "__main__":
    main()
