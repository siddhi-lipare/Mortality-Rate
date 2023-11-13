import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# Load the data
X = pd.read_csv("data/training_data.csv")
Y = pd.read_csv("data/training_data_targets.csv", header = None)

# Check for missing values
missing_values = X.isnull().sum()

missing_values_count = missing_values[missing_values > 0]
total_missing_values = missing_values_count.sum()

print(missing_values_count)
print("Total Missing Values:", total_missing_values)

# Checking the shapes of training data and targets
X_shape = X.shape
Y_shape = Y.shape

print("Training data Shape: ", X_shape)
print("Training targets Shape: ", Y_shape)

mean_impute = SimpleImputer(strategy = 'mean')
X_mean_impute = pd.DataFrame(data = mean_impute.fit_transform(X), columns = X.columns)
# print("After Mean Imputation: ")
# print(X_mean_impute.shape[0] - X_mean_impute.count())


X_train, X_test, y_train, y_test = train_test_split(
    X_mean_impute, Y.values.ravel(),
    test_size=0.2, random_state=42
)

# Define the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [80, 90, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 3, 4]
}

# Create the GridSearchCV object
scorer = make_scorer(f1_score, average = "weighted")
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring=scorer)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
f1_score_final = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'F1- Score of the Best Model: {f1_score_final:.2f}')
print(f'Accuracy of the Best Model: {accuracy:.2f}')