# Pritpal-Singh-X-Publiq

## Product Sales Prediction

This repository contains a Jupyter Notebook for predicting monthly sales of products using various machine learning models. The dataset consists of 500+ products from Amazon with features such as product name, MRP, price, star ratings, and number of ratings. The main tasks are to extract brand names from the product names and predict monthly sales using regression models.

## Steps and Explanation

### 1. Import Libraries

We begin by importing the necessary libraries for data processing, machine learning, and visualization.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### 2. Load and Preprocess Data

We load the data from a CSV file into a pandas DataFrame and preprocess it.

```python
df = pd.read_csv('product_data.csv')
```

### Data Cleaning & EDA
Cleaning numerical columns: remove commas and convert to float.
Handling missing values.

### 3. Train and Evaluate Models

We define and train multiple regression models, evaluate their performance, and store the results.

```python
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }

results_df = pd.DataFrame(results).T
results_df
```
Train multiple regression models, evaluate their performance, and store the results in a DataFrame.


### 4. Visualize Results with Heatmaps

Visualize the correlation matrix of features and the evaluation metrics of models using heatmaps. Identify the best model based on the R-squared value.

We visualize the correlation matrix of features and the evaluation metrics of models using heatmaps.


![image](https://github.com/pritpalcodes/Pritpal-Singh-X-Publiq/assets/90276050/9fa69f15-9b19-4f70-a0f5-f50ff9af8f36)

![image](https://github.com/pritpalcodes/Pritpal-Singh-X-Publiq/assets/90276050/8ec2c107-a533-425f-92d5-d165727f0952)


### 5. Save Predictions from Best Model

We use the best model to make predictions on the entire dataset and save the results to a CSV file.

```python
best_model = models[best_model_name]
df['Predicted Monthly Sales'] = best_model.predict(features)

df.to_csv('enhanced_product_data.csv', index=False)
df.head()
```

### Summary

1. **Data Loading and Preprocessing**: Extract brand names, clean numerical columns, handle missing values, and estimate monthly sales.
2. **Model Training and Evaluation**: Train multiple regression models and evaluate their performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.
3. **Visualization**: Use heatmaps to visualize the correlation between features and the performance metrics of the models.
4. **Prediction and Saving Results**: Use the best model to predict monthly sales for the entire dataset and save the predictions to a CSV file.

This notebook provides a comprehensive approach to data preprocessing, model training, evaluation, and visualization, helping to identify the best model for predicting product sales.

## Files in the Repository

- `publiq assignment.ipynb`: The main Jupyter Notebook with all the code.
- `Data_Pull.csv`: The dataset used for training and prediction.
- `final_enhanced_product_data.csv`: The output file with predicted monthly sales.
- `Pritpal Singh for Publiq.ipynb`: The Jupyter Notebook where where I did initial tinkering (original code).
 
## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage

1. Clone the repository.
2. Install the dependencies.
3. Open the `publiq assignment.ipynb` notebook.
4. Run the cells step-by-step to see the results.
