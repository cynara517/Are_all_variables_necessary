import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def run_grouped_correlation_analysis():

    DATA_FOLDER = "//"
    file_path = os.path.join(DATA_FOLDER, "xx.csv")
    output_dir = os.path.join(DATA_FOLDER, "xxx")
    os.makedirs(output_dir, exist_ok=True)


    df = pd.read_csv(file_path)
    X = df.iloc[:, :120]


    groups = {
        "Additive": [col for col in X.columns if "additive" in col.lower()],
        "Aryl_Halide": [col for col in X.columns if "aryl_halide" in col.lower()],
        "Ligand": [col for col in X.columns if "ligand" in col.lower()],
        "Base": [col for col in X.columns if "base" in col.lower()]
    }
        
       
        group_data = X[feature_list]
        corr_matrix = group_data.corr(method='pearson')
        
       
        figsize_val = max(10, len(feature_list) * 0.6)
        plt.figure(figsize=(figsize_val, figsize_val * 0.8))
        
    
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap='RdBu_r', 
            center=0,
            annot=len(feature_list) < 30,
            fmt=".2f", 
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title(f"Internal Pearson Correlation: {group_name}", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
       
        img_path = os.path.join(output_dir, f"corr_{group_name}.png")
        plt.savefig(img_path, bbox_inches='tight', dpi=300)
        plt.close()
        
       
        corr_matrix.to_excel(os.path.join(output_dir, f"matrix_{group_name}.xlsx"))



if __name__ == "__main__":
    run_grouped_correlation_analysis()