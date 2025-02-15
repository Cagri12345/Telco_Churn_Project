# Customer Churn Prediction Project

📄 **Project Overview**  
This project uses machine learning techniques for data analysis and model development to predict customer churn for a telecommunications company. Customer churn is a critical factor for companies looking to improve customer loyalty and reduce churn rates. This project focuses on churn prediction for telecommunications companies, utilizing various machine learning techniques and model optimizations.

📊 **Dataset Description**  
The dataset contains information about 7043 customers of a fictional telecommunications company based in California. The features in the dataset are as follows:

- **Customer Information**: CustomerId, Gender, SeniorCitizen, Partner, Dependents
- **Service Usage**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Information**: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target Variable**: Churn (Indicates if the customer has left: 1 - Left, 0 - Stayed)

⚙️ **Project Workflow**

1. **Data Exploration and Analysis**
   - **Summary Statistics and Visualizations**: Distribution and correlations of key features were examined.
   - **Missing and Outlier Values**: Missing and outlier values in the dataset were detected and handled appropriately.
   - **Correlation Analysis**: Relationships between features were analyzed to identify important variables.

2. **Data Preprocessing**
   - **Handling Missing Values**: Missing values in the TotalCharges variable were filled using the MonthlyCharges variable.
   - **Outlier Detection and Handling**: Outliers were identified and processed accordingly.
   - **Feature Scaling**: The data was scaled using StandardScaler for better processing.
   - **Categorical Variable Encoding**: Categorical variables like Gender, InternetService, and Contract were encoded using One-Hot Encoding.

3. **Model Development**
   - Various machine learning models were used to predict customer churn:
   
   **Base Model Training**:
   - Logistic Regression (LR): accuracy: 0.8046
   - KNN: accuracy: 0.7733
   - SVC: accuracy: 0.8001
   - CART: accuracy: 0.7338
   - Random Forest (RF): accuracy: 0.7894
   - Adaboost: accuracy: 0.7948
   - GBM: accuracy: 0.8012
   - LightGBM: accuracy: 0.7933
   - CatBoost: accuracy: 0.7981

4. **Hyperparameter Optimization**
   - Hyperparameter optimization was performed for different models. The hyperparameter ranges and results are as follows:

   ```python
   knn_params = {"n_neighbors": range(2, 50)}
   cart_params = {'max_depth': range(1, 20), "min_samples_split": range(2, 30)}
   rf_params = {"max_depth": [8, 15, None], "max_features": [5, 7, "auto"], "min_samples_split": [15, 20], "n_estimators": [200, 300]}
   xgboost_params = {"learning_rate": [0.1, 0.01], "max_depth": [5, 8], "n_estimators": [100, 200]}
   lightgbm_params = {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500]}

5. **Voting Classifier Model Training**
   - A Voting Classifier was created by combining base models, achieving higher performance with the following results:
   
   - **Accuracy**: 0.8028
   - **F1-Score**: 0.5840
   - **ROC-AUC**: 0.8449

6. **Model Evaluation**
   - The model’s performance was evaluated using Accuracy, F1-Score, and ROC-AUC. The Voting Classifier model achieved the best results:
   
   - **Accuracy**: 0.8028
   - **F1-Score**: 0.5840
   - **ROC-AUC**: 0.8449

📦 **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**:
  - pandas: Data manipulation
  - numpy: Numerical computations
  - scikit-learn: Machine learning algorithms
  - xgboost: XGBoost algorithm
  - lightgbm: LightGBM algorithm
  - catboost: CatBoost algorithm
  - matplotlib, seaborn: Data visualization

