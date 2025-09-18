# Disease-Prediction-System-using-Machine-Learning

A Disease Prediction System using Machine Learning analyzes patient data such as symptoms, history, and lab results to predict disease likelihood. By leveraging algorithms like Random Forest, SVM, or deep learning, it improves diagnostic accuracy, enables early detection, and supports doctors in making timely healthcare decisions.

---

## Libraries and Tools

### Core
- `numpy`, `pandas`, `scipy` → numerical computation and data manipulation  

### Visualization
- `matplotlib`, `seaborn`, `plotly` → data exploration and visual insights  

### Machine Learning
- `scikit-learn` → preprocessing, baseline models (Random Forest, SVM, Logistic Regression)  
- `xgboost`, `lightgbm` → advanced boosting algorithms for improved accuracy  

### NLP/Text (future extension)
- `nltk`, `spacy` → preprocessing and analysis of unstructured medical text  

### Model Persistence
- `joblib`, `pickle5` → saving and loading trained models  

### Utilities
- `tqdm` → progress bars  
- `ipython`, `jupyterlab` → interactive development  

### Data Preprocessing & Feature Encoding
- `pandas` → loading CSV files and handling tabular data  
- `sklearn.preprocessing.LabelEncoder` → converting ordinal categorical variables to integer labels  
- `sklearn.preprocessing.OneHotEncoder` or `pd.get_dummies` → transforming nominal categorical variables into one-hot encoded vectors  
- Encoded datasets saved as new CSV files (e.g., `dataset_encoded.csv`, `Training_encoded.csv`)  
- Ensures all categorical features are represented as numerical data compatible with machine learning models  

---

## Experimental Results

All models achieved perfect evaluation metrics on the dataset:

- **RandomForest**: `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **DecisionTree**: `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **Logistic Regression**: `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **SVM**: `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  
- **XGBoost**: `{'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}`  

---

## Model Justification

Although all models performed equally well, **Random Forest** was selected as the final model due to the following reasons:

- **Robustness & Stability** – Reduces variance by averaging multiple decision trees, minimizing overfitting.  
- **Feature Importance** – Provides insights into which symptoms contribute most to predictions.  
- **Generalization Ability** – Tends to perform reliably on unseen data, critical for healthcare applications.  
- **Scalability** – Efficiently handles large datasets with mixed data types.  
- **Proven Effectiveness** – Widely adopted in medical prediction tasks for its balance of accuracy and interpretability.  
