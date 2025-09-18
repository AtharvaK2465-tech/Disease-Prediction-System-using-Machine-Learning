# Disease Prediction System using Machine Learning

## Dataset Summary

- **Source:** Structured dataset containing symptoms and corresponding diseases.  
- **Features:** Symptom attributes (categorical).  
- **Target:** Disease classification (multi-class).  
- **Size:** (Rows × Columns – add specifics if available).  
- **Goal:** Build a predictive model that classifies diseases based on patient symptoms.  

---

## Data Preprocessing

To ensure compatibility with ML algorithms, categorical features were converted into numerical form:

- **Label Encoding** – Converted ordinal categorical variables into integer labels.  
- **One-Hot Encoding** – Transformed nominal categorical variables into binary vectors.  
- **Cleaned & Saved Datasets** – Produced `dataset_encoded.csv` and `Training_encoded.csv`.  

This preprocessing step ensures that all symptom-based features are machine-readable and consistent across experiments.  

---

## Model Selection & Training

Several machine learning models were trained and compared:

- **Logistic Regression** – Interpretable baseline.  
- **Decision Tree** – Simple and transparent.  
- **Support Vector Machine (SVM)** – Effective for high-dimensional data.  
- **Random Forest** – Ensemble of trees for robust predictions.  
- **XGBoost** – Gradient boosting for enhanced accuracy.  

---

## Libraries and Tools

### Core Libraries
- `numpy`, `pandas`, `scipy` → numerical computation, tabular data handling  
- `os`, `sys`, `warnings` → system utilities and error handling  

### Visualization
- `matplotlib`, `seaborn` → exploratory data analysis (EDA), correlation heatmaps, distribution plots  

### Data Preprocessing
- `scikit-learn.preprocessing` → `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`  
- `scikit-learn.compose` → `ColumnTransformer` for combining numeric & categorical pipelines  
- `scikit-learn.pipeline` → `Pipeline` for reproducible workflows  

### Model Selection & Evaluation
- `scikit-learn.model_selection` → `train_test_split`, `cross_val_score`, `KFold`, `GridSearchCV`, `RandomizedSearchCV`  
- `scikit-learn.metrics` → `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `ConfusionMatrixDisplay`  

### Machine Learning Models
- `DecisionTreeClassifier` (baseline decision tree)  
- `LogisticRegression` (linear baseline)  
- `RandomForestClassifier` (final model – robust ensemble)  
- `SVC` (Support Vector Machine with kernel tricks)  
- `XGBClassifier` (XGBoost gradient boosting)  

### Model Persistence
- `joblib` → save and reload trained pipelines (`.joblib` models)  

### Experiment Tracking (Outputs)
- `outputs/eda` → target distribution, correlation plots  
- `outputs/confusion_matrices` → confusion matrices per model  
- `outputs/models` → saved models (`RandomForest.joblib`, `best_model.joblib`, etc.)  
- `outputs/model_comparison.csv` → cross-validation & test metrics comparison  

---

## Experimental Results

All models achieved perfect evaluation metrics on the dataset:

- **RandomForest:** `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **DecisionTree:** `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **LogisticRegression:** `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **SVM:** `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **XGBoost:** `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  

---

## Model Justification

Although all models performed equally well, **Random Forest** was selected as the final model due to the following reasons:

- **Robustness & Stability** – Reduces variance by averaging multiple decision trees, minimizing overfitting.  
- **Feature Importance** – Provides insights into which symptoms contribute most to predictions.  
- **Generalization Ability** – Tends to perform reliably on unseen data, critical for healthcare applications.  
- **Scalability** – Efficiently handles large datasets with mixed data types.  
- **Proven Effectiveness** – Widely adopted in medical prediction tasks for its balance of accuracy and interpretability.  

---

## Conclusion

This project successfully developed a **Disease Prediction System** capable of predicting diseases with high accuracy using patient symptoms.  

- All models reached perfect metrics, but **Random Forest** was chosen as the final model for its reliability and interpretability.  
- The system demonstrates the potential of machine learning in healthcare to improve diagnostic accuracy and enable early intervention.  

---

## Project Description

**Disease-Prediction-System-using-Machine-Learning**  
A Disease Prediction System using Machine Learning analyzes patient data such as symptoms, history, and lab results to predict disease likelihood. By leveraging algorithms like Random Forest, SVM, or deep learning, it improves diagnostic accuracy, enables early detection, and supports doctors in making timely healthcare decisions.  
