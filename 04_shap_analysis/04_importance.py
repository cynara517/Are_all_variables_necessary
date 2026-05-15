import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor


# ==========================================
data_dir = "//"
out_dir = "xx"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


data_path = os.path.join(data_dir, "scaled.csv") 
target_node = "yield"

#Out-of-sample extrapolation
TRAIN_IDS = []
TEST_IDS  = []

nmr_shift_to_additive = {
    -0.958351938: 2, 0.785761996: 4, 0.304570487: 6, -1.354144272: 1,
    0.882657214: 3, 0.491791756: 5, -1.10287362: 8, -0.012391838: 10,
    0.61989052: 12, -0.966563397: 14, 0.445807585: 9, -0.53135606: 11,
    -1.171849877: 13, -1.626764717: 15, 2.321304866: 23, 0.820250124: 17,
    1.25381517: 19, 0.572264057: 21, -1.051962573: 16, -0.86145672: 18,
    0.478653422: 20, 0.660947816: 22
}

PARAMS_SPECIFIED = {
    "RandomForest": {
        "30":  {'n_estimators': 257, 'max_depth': 21, 'random_state': 42, 'n_jobs': -1},
        "120": {'n_estimators': 498, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
    },
    "CatBoost": {
        "30":  {'iterations': 901, 'depth': 8, 'learning_rate': 0.08682345477771439, 'silent': True, 'random_state': 42},
        "120": {'iterations': 676, 'depth': 8, 'learning_rate': 0.1205373476195167, 'silent': True, 'random_state': 42}
    },
    "SVM": {
        "30":  {'C': 73.8402955007617, 'gamma': 0.056447093803982176},
        "120": {'C': 83.33164129953337, 'gamma': 0.01424481178907202}
    }
}

nodes_30 = [
    'additive_surface_area', 'additive_V1_intensity', 'additive_.O1_electrostatic_charge', 
    'additive_.C4_NMR_shift', 'additive_E_LUMO', 'additive_V1_frequency', 
    'additive_.C3_electrostatic_charge', 'additive_dipole_moment', 'additive_.C4_electrostatic_charge', 
    'additive_.C5_NMR_shift', 'additive_.C5_electrostatic_charge', 'additive_hardness', 
    'additive_electronegativity', 'aryl_halide_ovality', 'aryl_halide_.H3_electrostatic_charge', 
    'aryl_halide_V3_frequency', 'aryl_halide_V3_intensity', 'aryl_halide_dipole_moment', 
    'aryl_halide_molecular_weight', 'aryl_halide_.C1_NMR_shift', 'aryl_halide_V1_intensity', 
    'aryl_halide_.C2_NMR_shift', 'aryl_halide_V2_intensity', 'aryl_halide_.C3_NMR_shift', 
    'aryl_halide_.C4_NMR_shift', 'base_surface_area', 'base_electronegativity', 
    'base_.N1_electrostatic_charge', 'ligand_.C10_NMR_shift', 'ligand_V3_frequency'
]


COLOR_REDUNDANT = "#757575"  
COLOR_ADDITIVE  = "#D32F2F"  
COLOR_ARYL      = "#1976D2"  
COLOR_BASE      = "#388E3C"  
COLOR_LIGAND    = "#7B1FA2"  


custom_cmap = mcolors.LinearSegmentedColormap.from_list("cyan_magenta", ["#00BFFF", "#FF00FF"])


# ==========================================
print("Loading real dataset and mapping Additive IDs...")
df = pd.read_csv(data_path)


def get_additive_id(val):
    closest_key = min(nmr_shift_to_additive.keys(), key=lambda k: abs(k - val))
    return nmr_shift_to_additive[closest_key]


df['additive_id'] = df['additive_.C3_NMR_shift'].apply(get_additive_id)


train_mask = df['additive_id'].isin(TRAIN_IDS)
test_mask  = df['additive_id'].isin(TEST_IDS)

y_train = df[train_mask][target_node].values
y_test  = df[test_mask][target_node].values


X_120_train = df[train_mask].drop(columns=[target_node, 'additive_id'])
X_120_test  = df[test_mask].drop(columns=[target_node, 'additive_id'])


X_30_train = X_120_train[nodes_30]
X_30_test  = X_120_test[nodes_30]


scaler_120 = StandardScaler().fit(X_120_train)
X120_train_sc = pd.DataFrame(scaler_120.transform(X_120_train), columns=X_120_train.columns)
X120_test_sc  = pd.DataFrame(scaler_120.transform(X_120_test), columns=X_120_test.columns)

scaler_30 = StandardScaler().fit(X_30_train)
X30_train_sc = pd.DataFrame(scaler_30.transform(X_30_train), columns=X_30_train.columns)
X30_test_sc  = pd.DataFrame(scaler_30.transform(X_30_test), columns=X_30_test.columns)


# ==========================================
def draw_shap_plot(model_name, var_type):
    print(f"--> Generating plot for {model_name} ({var_type} variables)...")
    
    X_tr = X120_train_sc if var_type == "120" else X30_train_sc
    X_te = X120_test_sc if var_type == "120" else X30_test_sc
    params = PARAMS_SPECIFIED[model_name][var_type]
    
    if model_name == "RandomForest":
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        X_eval = X_te
        
    elif model_name == "CatBoost":
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        X_eval = X_te
        
    elif model_name == "SVM":
        model = SVR(**params)
        model.fit(X_tr, y_train)
        background = shap.kmeans(X_tr, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        X_eval = shap.sample(X_te, 150, random_state=42) if len(X_te) > 150 else X_te
        shap_values = explainer.shap_values(X_eval)

    fig, ax = plt.subplots(figsize=(8, 6))
    

    shap.summary_plot(
        shap_values, X_eval, 
        max_display=10, 
        plot_type="violin", 
        cmap=custom_cmap, 
        show=False, 
        color_bar=False 
    )
    
    ax = plt.gca()
    ax.set_title("") 
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
        
    ax.set_xlabel("SHAP Value", fontsize=13, fontweight='medium', labelpad=10)
    ax.axvline(x=0, color='#A9A9A9', linestyle='--', linewidth=1.2, zorder=0)
    ax.tick_params(axis='x', labelsize=11, bottom=True, length=4, color='#A9A9A9')
    ax.tick_params(axis='y', length=0) 
    

    for tick in ax.get_yticklabels():
        var_name = tick.get_text()
        if var_name not in nodes_30:
            tick.set_color(COLOR_REDUNDANT)
            tick.set_fontweight('normal')
        else:
            tick.set_fontweight('bold')
            if 'additive' in var_name.lower():
                tick.set_color(COLOR_ADDITIVE)
            elif 'aryl_halide' in var_name.lower():
                tick.set_color(COLOR_ARYL)
            elif 'base' in var_name.lower():
                tick.set_color(COLOR_BASE)
            elif 'ligand' in var_name.lower():
                tick.set_color(COLOR_LIGAND)
            else:
                tick.set_color('black')
                

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, aspect=20, pad=0.05)
    cb.set_ticks([]) 
    cb.outline.set_visible(False) 
    
    cb.set_label("Feature Value", size=12, labelpad=5)
    cb.ax.text(0.5, 1.02, 'High', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#FF00FF')
    cb.ax.text(0.5, -0.02, 'Low', transform=cb.ax.transAxes, ha='center', va='top', fontsize=11, fontweight='bold', color='#00BFFF')

    # 导出文件
    out_pdf = os.path.join(out_dir, f"SHAP_Extrapolation_{model_name}_{var_type}Vars.pdf")
    out_svg = os.path.join(out_dir, f"SHAP_Extrapolation_{model_name}_{var_type}Vars.svg")
    
    plt.savefig(out_pdf, format="pdf", bbox_inches='tight')
    plt.savefig(out_svg, format="svg", bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} | {out_svg}")


# ==========================================
target_models = ["RandomForest", "CatBoost", "SVM"]
target_vars = ["120", "30"]

for m in target_models:
    for v in target_vars:
        draw_shap_plot(m, v)

print("\n🎉 Extrapolation SHAP plots on real data successfully generated!")