#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


data_file = "//"
target_node = "yield"
train_ratios = [0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.025]


PARAMS_SPECIFIED = {
    "RandomForest": {"30": {'n_estimators': 257, 'max_depth': 21, 'random_state': 42}, "120": {'n_estimators': 498, 'max_depth': 20, 'random_state': 42}},
    "CatBoost": {"30": {'iterations': 901, 'depth': 8, 'learning_rate': 0.086, 'silent': True, 'random_state': 42}, "120": {'iterations': 676, 'depth': 8, 'learning_rate': 0.120, 'silent': True, 'random_state': 42}},
    "LightGBM": {"30": {'n_estimators': 971, 'num_leaves': 32, 'learning_rate': 0.071, 'verbose': -1, 'random_state': 42}, "120": {'n_estimators': 926, 'num_leaves': 27, 'learning_rate': 0.092, 'verbose': -1, 'random_state': 42}},
    "SVM": {"30": {'C': 73.84, 'gamma': 0.056}, "120": {'C': 83.33, 'gamma': 0.014}},
    "KRR": {"30": {'alpha': 0.004, 'kernel': 'rbf', 'gamma': 0.015}, "120": {'alpha': 0.001, 'kernel': 'rbf', 'gamma': 0.002}},
    "RBF_Net": {"30": {'alpha': 0.002, 'gamma': 0.015}, "120": {'alpha': 0.0001, 'gamma': 0.001}}
}


df_raw = pd.read_csv(data_file)
nodes_30 = ['additive_surface_area', 'additive_V1_intensity', 'additive_.O1_electrostatic_charge', 'additive_.C4_NMR_shift', 'additive_E_LUMO', 'additive_V1_frequency', 'additive_.C3_electrostatic_charge', 'additive_dipole_moment', 'additive_.C4_electrostatic_charge', 'additive_.C5_NMR_shift', 'additive_.C5_electrostatic_charge', 'additive_hardness', 'additive_electronegativity', 'aryl_halide_ovality', 'aryl_halide_.H3_electrostatic_charge', 'aryl_halide_V3_frequency', 'aryl_halide_V3_intensity', 'aryl_halide_dipole_moment', 'aryl_halide_molecular_weight', 'aryl_halide_.C1_NMR_shift', 'aryl_halide_V1_intensity', 'aryl_halide_.C2_NMR_shift', 'aryl_halide_V2_intensity', 'aryl_halide_.C3_NMR_shift', 'aryl_halide_.C4_NMR_shift', 'base_surface_area', 'base_electronegativity', 'base_.N1_electrostatic_charge', 'ligand_.C10_NMR_shift', 'ligand_V3_frequency']
nodes_30 = [f for f in nodes_30 if f in df_raw.columns]
nodes_120 = (nodes_30 + [c for c in df_raw.columns if c not in nodes_30 and c != target_node])[:120]

df_pool, df_test_fixed = train_test_split(df_raw, test_size=0.30, random_state=42)

def get_model(model_name, params):
    if model_name == "RandomForest": return RandomForestRegressor(**params)
    if model_name == "CatBoost": return CatBoostRegressor(**params)
    if model_name == "LightGBM": return LGBMRegressor(**params)
    if model_name == "SVM": return SVR(**params)
    if model_name in ["KRR", "RBF_Net"]: return KernelRidge(alpha=params.get('alpha', 0.1), kernel='rbf', gamma=params.get('gamma', 0.1))
    return None


all_results = []
for ratio in train_ratios:
    current_frac = ratio / 0.7 
    df_train_sub = df_pool.sample(frac=min(current_frac, 1.0), random_state=42)
    for model_name, configs in PARAMS_SPECIFIED.items():
        for tag, features in [("30", nodes_30), ("120", nodes_120)]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train_sub[features])
            X_test = scaler.transform(df_test_fixed[features])
            y_train, y_test = df_train_sub[target_node].values, df_test_fixed[target_node].values
            
            model = get_model(model_name, configs[tag])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            all_results.append({
                "Ratio": ratio, "Model": model_name, "Feats": tag,
                "R2": r2_score(y_test, preds), "RMSE": np.sqrt(mean_squared_error(y_test, preds))
            })

pd.DataFrame(all_results).to_csv("learning_curve_data.csv", index=False)
print("Data saved to learning_curve_data.csv")