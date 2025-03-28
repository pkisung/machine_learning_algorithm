# machine_learning_algorithm
# Abalone Dataset - Machine Learning Model Comparison

## Overview
This project compares the performance of multiple machine learning models on the Abalone dataset from Kaggle. The goal is to predict the age of abalones based on physical characteristics using various regression models.

## Models Evaluated
- **K-Nearest Neighbors (KNN)**
- **Multi-Layer Perceptron (MLP)**
- **Random Forest**
- **Decision Tree**
- **Artificial Neural Network (ANN) Regression**

## Dataset
The Abalone dataset consists of physical measurements of abalones, which are used to estimate their age.

## Performance Metrics
The models were evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R2 Score**: Indicates how well the model explains the variance in the data (closer to 1 is better).
- **Root Mean Squared Error (RMSE)**: Square root of MSE, providing error in original units.
- **Relative Mean Absolute Error (RMAE)**: Measures MAE relative to the mean target value.

## Results
| Model                  | MAE     | MSE     | RMSE   | R2 Score |
|------------------------|---------|--------|---------|----------|
| K-Nearest Neighbors    | 1.447   | 4.136  | 2.034   | 0.4898   |
| Multi-Layer Perceptron | 2.113   | 7.760  | 2.786   | 0.0428   |
| Random Forest          | 1.429   | 1.981  | 1.981   | 0.5211   |
| Decision Tree          | 1.870   | 2.620  | 2.620   | 0.15     |
| ANN Regression         | 0.139   | 1.922  | 1.922   | 0.544    |

(Note: `N/A` indicates that results for these models were not successfully extracted.)

## Conclusion
- The ANN Regression model achieved an R2 score of **0.544**, indicating moderate predictive power.
- The Decision Tree model showed slightly lower MAE compared to Random Forest but had a higher MSE.
- Further hyperparameter tuning and feature engineering could improve performance.
- The comparison provides insights into the effectiveness of different models for this dataset.

## How to Use
To run the models, use the respective Jupyter notebooks:
1. `preprocessing218_knn.ipynb` - KNN Model
2. `preprocessing218(MLP).ipynb` - MLP Model
3. `RandomForest218.ipynb` - Random Forest Model
4. `Decision_tree_218.ipynb` - Decision Tree Model
5. `ANN_Regression.ipynb` - ANN Regression Model

## Future Work
- Extract results for KNN and MLP models.
- Experiment with hyperparameter tuning for better results.
- Implement ensemble methods to combine models.
- Use additional feature engineering techniques to improve prediction accuracy.

