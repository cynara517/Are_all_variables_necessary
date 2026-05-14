# Are all variables necessary? Reshaping the C-N cross-coupling yield prediction paradigm

This repository contains scripts for a four-step information decoupling workflow designed to reduce 120 DFT descriptors to 30 key variables, significantly reshaping the yield prediction paradigm for Buchwald-Hartwig amination reactions. Alongside the feature selection workflow, this repository includes scripts for machine learning modeling, hierarchical clustering, and SHAP analysis, as well as the Excel spreadsheets with the initial descriptor matrices and input data. 

This workflow refers to our paper currently under review.

### Dependencies

This workflow relies on published computational tools, including [scikit-learn](1),  [Euclidean spatial distance](2), [Pearson correlation](3), [Ridge regression](4), [MIC](5) and [SHAP](6). See the links for detailed installation guides and documentation.

---

### Step 1: Dataset and Feature Space Definition

The dataset utilized in this study comprises 4,608 experimental data points for Buchwald–Hartwig amination reactions, originally reported by Ahneman *et al.*⁶ The raw input data is located in the `data/raw/` folder.

Each reaction instance is characterized within a high-dimensional feature space consisting of 120 DFT descriptors. These descriptors are systematically categorized into three classes (molecular, atomic, and vibrational descriptors). 

> **Note on Feature Generation:** The initial generation of these 120 DFT descriptors was performed using protocols adapted from the work of `[Insert Collaborator's Name or Your Group's Previous Work, e.g., Dr. X / X Lab]`. Their base feature extraction scripts can be found on their GitHub repository `[Insert Link to Collaborator's GitHub]`. 

---

### Step 2: Information Decoupling and Feature Selection

To decouple redundant information and extract essential features, a four-stage variable screening paradigm was developed. The scripts are located in the `scripts/01_information_decoupling/` directory.

1.  **Correlation Identification:** Calculates pairwise Pearson correlation coefficients (|r|). 
2.  **Dependency Deconstruction:** Utilizes Ridge regression to map hierarchical relational chains. *This script (`2_dependency_deconstruction.py`) was adapted from the linear deconstruction code originally written by `[Insert Collaborator/Co-author Name]` and has been previously discussed in `[Insert Reference/Link to previous paper]`. Mofidications were made to specifically handle the independent reactant categories.*
3.  **Representative Feature Selection:** Quantifies the Mutual Information (MI) contribution.
4.  **Dimensionality Reduction:** Selects the highest informational contributor for each chain.

This systematic reduction resulted in **30 essential key variables**, effectively eliminating 75% of the original redundancy.
### Step 3: Machine Learning Architecture and Bayesian Optimization

The `scripts/02_machine_learning/` folder contains the Python scripts used to evaluate predictive performance using six representative regression algorithms (Random Forest, CatBoost, LightGBM, SVM, KRR, RBF).

Following a 7:3 train-test split protocol, hyperparameter configurations for both the 120-variable and 30-variable systems were independently optimized via Bayesian optimization within the Optuna framework.

To ensure reproducibility, models were evaluated via a 5-fold cross-validation scheme.

---

### Step 4: Hierarchical Clustering

Additive clustering based on Euclidean distance quantification was performed using the script `scripts/03_hierarchical_clustering/clustering_analysis.py`. This analysis helps in understanding the structural and chemical similarities among the additives in the decoupled feature space.

---

### Step 5: SHAP Analysis

To extract mechanistic insights and interpret the decision-making process of our optimized tree-based models, SHAP (SHapley Additive exPlanations) values were calculated. The analysis is located in `scripts/04_shap_analysis/`. This implementation builds upon the original game-theoretic framework and `shap` library developed by Lundberg and Lee.⁵

---

### References

1. Pedregosa, F. *et al.* Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011). [Link](https://scikit-learn.org/)
2. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. Optuna: A Next-generation Hyperparameter Optimization Framework. In *KDD* (2019). [Link](https://optuna.org/)
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. CatBoost: unbiased boosting with categorical features. *Advances in neural information processing systems*, 31 (2018). [Link](https://catboost.ai/)
4. Ke, G. *et al.* LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30 (2017). [Link](https://lightgbm.readthedocs.io/en/latest/)
5. Lundberg, S. M. & Lee, S.-I. A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems* **30** (2017). [Link](https://shap.readthedocs.io/)
6. Ahneman, D. K., Estrada, J. G., Lin, S., Dreher, S. D. & Doyle, A. G. Predicting reaction performance in C–N cross-coupling using machine learning. *Science* **360**, 186–190 (2018).
7. `[Please Add Your Collaborator's Base Protocol Paper Here if applicable]`
