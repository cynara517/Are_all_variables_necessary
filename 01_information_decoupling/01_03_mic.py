import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import os

def run_mi_analysis():

    DATA_FOLDER = "//"
    file_path = os.path.join(DATA_FOLDER, "xx")
    output_dir = os.path.join(DATA_FOLDER, "mi_analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    # 假设前120列是特征，yield是目标
    X = df.iloc[:, :120]
    y = df['yield']


    # discrete_features=False
    mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    

    mi_results = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_results = mi_results.sort_values(ascending=False)


    plt.figure(figsize=(10, 12))
    mi_results.head(30).sort_values().plot(kind='barh', color='teal')
    plt.title("Top 30 Features by Mutual Information with Yield")
    plt.xlabel("Mutual Information Score")
    

    plt.savefig(os.path.join(output_dir, "mi_scores_top30.png"), bbox_inches='tight', dpi=150)
    mi_results.to_excel(os.path.join(output_dir, "mi_scores_full.xlsx"))


    return mi_results

if __name__ == "__main__":
    mi_rank = run_mi_analysis()