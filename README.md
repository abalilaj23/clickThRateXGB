# Comparing Prediction Models for Click-Through Rate (CTR) Prediction

## Overview:
This project compares several machine learning models for predicting Click-Through Rate (CTR) based on various ad and query characteristics. The models evaluated include Linear Regression, Decision Tree Regression (CART), Random Forest, and XGBoost. Performance metrics such as Out-of-Sample R² (OSR²), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are used to assess the effectiveness of each model. Furthermore, XGBoost is tuned using both Grid Search and Random Search to optimize its hyperparameters.

## 1. Packages and Data
This project relies on several Python libraries:
- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For machine learning models, metrics, and utilities.
- **XGBoost**: For gradient boosting models.

The dataset consists of a training set (`train.csv`) and a test set (`test.csv`). The columns include features like `titleWords`, `adWords`, `depth`, `position`, as well as target variables like CTR.

## 2. Data Preprocessing
- **Categorical Features**: The gender and age columns are categorical. They are converted to dummy variables using `pd.get_dummies()`, which converts these into binary columns.
- **Target Variable**: The target variable, CTR, is separated from the rest of the features and stored in `y_train` and `y_test`.

## 3. Modeling and Evaluation
The following models are built and evaluated:

### Model 1: Linear Regression
- **Performance Metrics**:
    - OSR²: 0.375
    - MAE: 0.036
    - RMSE: 0.060
- **Coefficients**: The model finds that features like `advCTR`, `advCTRInPos`, and `queryCTRInPos` have significant coefficients.

### Model 2: Decision Tree Regression (CART)
- **Initial Performance**:
    - OSR²: 0.084
    - MAE: 0.041
    - RMSE: 0.073
- **Improved Model**: After applying cost complexity pruning to optimize the tree, the model achieves:
    - OSR²: 0.376
    - MAE: 0.035
    - RMSE: 0.060
- **Tree Visualization**: A decision tree with a depth of 7 and 39 nodes is visualized.

### Model 3: Random Forest
- **Performance Metrics**:
    - OSR²: 0.468
    - MAE: 0.032
    - RMSE: 0.056
- **Summary**: The Random Forest model performs better than the Decision Tree but can potentially be further improved.

### Model 4: XGBoost
- **Performance Metrics**:
    - OSR²: 0.470
    - MAE: 0.032
    - RMSE: 0.055
- **Summary**: XGBoost gives a slight improvement over Random Forest but is further optimized using hyperparameter tuning.

## 4. Hyperparameter Optimization

### Grid Search
- **Parameters Tuned**:
    - `n_estimators`: 10, 50, 100
    - `max_depth`: 1, 2, 3, 4, 5
    - `learning_rate`: 0.001, 0.01, 0.1, 0.3
- **Best Hyperparameters**:
    - `learning_rate`: 0.1
    - `max_depth`: 4
    - `n_estimators`: 80
- **Performance Metrics for Best Estimator**:
    - OSR²: 0.477
    - MAE: 0.032
    - RMSE: 0.055

### Random Search
Random Search explores more parameter combinations, with results:
- **Best Hyperparameters**:
    - `n_estimators`: 660
    - `max_depth`: 3
    - `learning_rate`: 0.062
- **Performance Metrics**:
    - OSR²: 0.487
    - MAE: 0.032
    - RMSE: 0.055
- **Feature Importance**: The most influential features are displayed using `plot_importance` for the best XGBoost model.

## 5. Conclusion
The XGBoost model with Random Search hyperparameters yields the best performance, surpassing all other models evaluated. It shows the highest OSR² of 0.487, with MAE and RMSE metrics comparable to those of other models, but with slight improvements.

## 6. Visualizations
- **Decision Tree Visualization**: The pruned decision tree is visualized with feature names and color coding.
- **Feature Importance**: A plot of feature importance (gain) for the best XGBoost model highlights the most important features for CTR prediction.

## 7. Further Work
To further improve the model:
- Additional feature engineering or domain-specific insights could be incorporated.
- A more extensive hyperparameter search for XGBoost could be conducted to explore additional parameter settings.

## 8. Acknowledgments
- Scikit-learn and XGBoost libraries for the machine learning models and hyperparameter optimization tools.
- Matplotlib for visualizations.
