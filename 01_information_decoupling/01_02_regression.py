import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


data_path = "//"
df = pd.read_csv(data_path)


test_features = [

]

independent_nodes = [
  
]

def get_group(name):
    if name.startswith("ligand"): return "ligand"
    if name.startswith("additive"): return "additive"
    if name.startswith("aryl_halide"): return "aryl_halide"
    if name.startswith("base"): return "base"
    return "other"

output_data = []

for target in test_features:
    target_group = get_group(target)
    X_cols = [col for col in independent_nodes if get_group(col) == target_group]
    
    if not X_cols:
        continue
        
    X = df[X_cols]
    y = df[target]
    

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    r2 = r2_score(y, model.predict(X))
    
    #  Y = b + w1*X1 + w2*X2 ...
    intercept = model.intercept_
    coefs = model.coef_
    
    formula_terms = [f"{intercept:.4f}"]
    for val, name in zip(coefs, X_cols):
        sign = "+" if val >= 0 else "-"
        formula_terms.append(f"{sign} {abs(val):.4f} * [{name}]")
    
    formula_str = " ".join(formula_terms)
    
    output_data.append({
        "Variable_Y": target,
        "Variable_X": ", ".join(X_cols),
        "R2": f"{r2:.4f}",
        "Formula": formula_str
    })


output_df = pd.DataFrame(output_data)
output_file = "variable_reconstruction_results.csv"
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

print(output_df[["Variable_Y", "R2"]].head())