# Are all the variables necessary? A case study of 120 variables for C-N cross-coupling reaction


This repository contains scripts for a four-step information decoupling workflow designed to reduce 120 DFT descriptors to 30 key variables, significantly reshaping the yield prediction paradigm for Buchwald-Hartwig amination reactions. Alongside the feature selection workflow, this repository includes scripts for machine learning modeling, hierarchical clustering, and SHAP analysis, as well as the Excel spreadsheets with the initial descriptor matrices and input data. 

This workflow refers to our paper currently under review.

### Dependencies

This workflow relies on published computational tools, including [scikit-learn](1),  [Euclidean spatial distance](2), [Pearson correlation](3), [Ridge regression](4), [MIC](5) and [SHAP](6). See the links for detailed installation guides and documentation.

---

### Step 0: Dataset and Feature Space Definition

The dataset utilized in this study comprises 4,608 experimental data points for Buchwald–Hartwig amination reactions, originally reported by Ahneman*, Doyle et al.(1). The raw input data is located in the `data/` folder.

Each reaction instance is characterized within a high-dimensional feature space consisting of 120 DFT descriptors. These descriptors are systematically categorized into three classes (molecular, atomic, and vibrational descriptors). 

> **Note on Feature Generation:** The initial generation of these 120 DFT descriptors was performed using protocols adapted from the work of doylelab. Their base feature extraction scripts can be found on their GitHub repository https://github.com/doylelab/rxnpredict. 

---

### Step 1: Information Decoupling and Feature Selection

To decouple redundant information and extract essential features, a four-stage variable screening paradigm was developed. The scripts are located in the `scripts/01_information_decoupling/` directory.

1.  **Correlation Identification:** Calculates pairwise Pearson correlation coefficients (|r|). 
2.  **Dependency Deconstruction:** Utilizes Ridge regression to map hierarchical relational chains.
3.  **Representative Feature Selection:** Quantifies the Mutual Information (MI) contribution.
4.  **Dimensionality Reduction:** Selects the highest informational contributor for each chain.

This systematic reduction resulted in **30 essential key variables**, effectively eliminating 75% of the original redundancy.
### Step 2: Machine Learning Architecture and Bayesian Optimization

The `scripts/02_machine_learning/` folder contains the Python scripts used to evaluate predictive performance using six representative regression algorithms (Random Forest(7), CatBoost(8), LightGBM(9), SVM(10), KRR(11), RBF(12)).

Following a 7:3 train-test split protocol, hyperparameter configurations for both the 120-variable and 30-variable systems were independently optimized via Bayesian optimization(13) within the Optuna framework https://optuna.org/.

To ensure reproducibility, models were evaluated via a 5-fold cross-validation scheme.

---

### Step 3: Hierarchical Clustering

Additive clustering based on Euclidean distance quantification was performed using the script `scripts/03_hierarchical_clustering/`. This analysis helps in understanding the structural and chemical similarities among the additives in the decoupled feature space.

---

### Step 4: SHAP Analysis

To extract mechanistic insights and interpret the decision-making process of our optimized tree-based models, SHAP (SHapley Additive exPlanations) values were calculated. The analysis is located in `scripts/04_shap_analysis/`. This implementation builds upon the original game-theoretic framework https://github.com/shap/shap, https://shap.readthedocs.cn/en/latest/index.html.

---

### References
1.https://scikit-learn.org/stable/index.html
2.D. S. Broomhead, D. Lowe, Multivariable functional interpolation and adaptive networks. Complex Syst. 2, 321-355 (1988).
3.K. Pearson, Mathematical contributions to the theory of evolution.—III. Regression, heredity, and panmixia. Philos. Trans. R. Soc. London Ser. A 187, 253-318 (1896).
4.A. E. Hoerl, R. W. Kennard, Ridge regression: Biased estimation for nonorthogonal problems. Technometrics 12, 55-67 (1970).
5.C. E. Shannon, A mathematical theory of communication. Bell Syst. Tech. J. 27, 379-423
6.S. M. Lundberg, S. I. Lee, A unified approach to interpreting model predictions. Adv. Neural Inf. Process. Syst. 30, 4765-4774 (2017).
7.L. Breiman, Random forests. Mach. Learn. 45, 5-32 (2001).
8.L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin, CatBoost: unbiased boosting with categorical features. Adv. Neural Inf. Process. Syst. 31, 6639-6649 (2018).
9.G. Ke et al., LightGBM: A highly efficient gradient boosting decision tree. Adv. Neural Inf. Process. Syst. 30, 3146-3154 (2017).
10.C. Cortes, V. Vapnik, Support-vector networks. Mach. Learn. 20, 273-297 (1995).
11.B. E. Boser, I. M. Guyon, V. N. Vapnik, A training algorithm for optimal margin classifiers. Proc. 5th Annu. Workshop Comput. Learn. Theory (ACM, 1992), pp. 144-152.
12.C. Saunders, A. Gammerman, V. Vovk, Ridge regression learning algorithm in dual variables. Proc. 15th Int. Conf. Mach. Learn. (1998), pp. 515-521.
13.https://distill.pub/2020/bayesian-optimization/
