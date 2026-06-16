import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error

# data
data = pd.read_csv("ML_MODEL_LABELED_INPUT.csv", na_values=["NA", "null", "?", " "], engine="python")
print(data.columns)

# gene names
gene_names = data.iloc[:, 0].values

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

# input
X = data[features]

# RNA-seq expression values
y = data["ct0_rpkm_cm_avg"]

# class label column
class_col = 'label'

# train test split
X_train, X_test, y_train, y_test, genes_train, genes_test = train_test_split(X, y, gene_names, test_size=0.30, random_state=89)

# scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# log transform RNA expression
y_train_scaled = np.log1p(y_train)
y_test_scaled = np.log1p(y_test)

# model
model = RandomForestRegressor(n_estimators=200, min_samples_split=2, max_depth=None, random_state=999, n_jobs=-1)
model.fit(X_train_scaled, y_train_scaled)

# prediction
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test_scaled, y_pred)
mae = mean_absolute_error(y_test_scaled, y_pred)

print("R2:", r2)
print("MAE:", mae)



# -----SHAP analysis-----

explainer = shap.TreeExplainer(model)

output_dir = '/home/group_nithya01/Dharani/ml/location-specific/dataset2/pol_per/waterfall_plots/'
os.makedirs(output_dir, exist_ok=True)

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# # compute SHAP values for all test genes
# shap_values_all = explainer(X_test_scaled_df)

#
# # -----SHAP values dataframe------
#
# shap_df = pd.DataFrame(shap_values_all.values, columns=features)
# shap_df.insert(0, 'gene', genes_test)
# shap_df.insert(1, 'class', data.loc[X_test.index, class_col].values)
#
# shap_df.to_csv(f"{output_dir}/shap_value_all_genes.csv", index=False)

target_genes = [
    'NM_001359834',
    'NM_001347371',
    'NM_001113246',
    'NM_172563',
    'NM_001291026',
    'NM_001177995'
]

selected_indices = []
selected_gene_names = []

for gene in target_genes:
    idx = np.where(genes_test == gene)[0]

    if len(idx) == 0:
        print(f"{gene} not found in test set")
    else:
        selected_indices.append(idx[0])
        selected_gene_names.append(gene)

X_selected = X_test_scaled_df.iloc[selected_indices]

shap_values_selected = explainer(X_selected)

# waterfall plot
for i, gene in enumerate(selected_gene_names):
    print(f"Generating waterfall plot for {gene}")
    shap.plots.waterfall(shap_values_selected[i], show=False, max_display=20)
    plt.title(gene)
    plt.savefig(f"{output_dir}/{gene}_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

print("All waterfall plots generated")
