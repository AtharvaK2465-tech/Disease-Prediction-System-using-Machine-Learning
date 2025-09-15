# Disease-Prediction-System-using-Machine-Learning
A Disease Prediction System using Machine Learning analyzes patient data such as symptoms, history, and lab results to predict disease likelihood. By leveraging algorithms like Random Forest, SVM, or deep learning, It improves diagnostic accuracy, enables early detection, and supports doctors in making timely healthcare decisions.

## Libraries and Tools  

- **Core**  
  - `numpy`, `pandas`, `scipy` → numerical computation and data manipulation  

- **Visualization**  
  - `matplotlib`, `seaborn`, `plotly` → data exploration and visual insights  

- **Machine Learning**  
  - `scikit-learn` → preprocessing, baseline models (Random Forest, SVM, Logistic Regression)  
  - `xgboost`, `lightgbm` → advanced boosting algorithms for improved accuracy  

- **NLP/Text (future extension)**  
  - `nltk`, `spacy` → preprocessing and analysis of unstructured medical text  

- **Model Persistence**  
  - `joblib`, `pickle5` → saving and loading trained models  

- **Utilities**  
  - `tqdm` (progress bars), `ipython`, `jupyterlab` (interactive development)  

- **Data Preprocessing**
- **Categorical Feature Encoding**
  - `pandas` → loading CSV files and handling tabular data  
  - `sklearn.preprocessing.LabelEncoder` → converting ordinal categorical variables to integer labels  
  - `sklearn.preprocessing.OneHotEncoder` or `pd.get_dummies` → transforming nominal categorical variables into one-hot encoded vectors  
  - Encoded datasets saved as new CSV files (e.g., `dataset_encoded.csv`, `Training_encoded.csv`)  
  - Ensures all categorical features are represented as numerical data compatible with machine learning models

## Disease Prediction Models

- **Logistic Regression**
  - Simple, interpretable model that estimates disease probability using linear relationships.
  - Fast to train and provides transparent results, useful for clinical decision-making.
  - Best suited when features have approximately linear influence on outcomes.

- **Support Vector Machine**
  - Effective in high-dimensional spaces and models non-linear boundaries with kernels.
  - Often yields strong accuracy but requires careful tuning and is less interpretable.
  - Suitable for complex symptom datasets where capturing intricate patterns is key.

- **Random Forest**
  - Ensemble of decision trees that handles non-linear relationships and mixed feature types.
  - Robust to noise and overfitting, provides feature importance insights.
  - Ideal for heterogeneous, noisy healthcare data requiring high predictive power.
