import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

# Load the data
training_data = pd.read_csv("data/training_data.csv")
training_targets = pd.read_csv("data/training_data_targets.csv")
merged_training_data = pd.concat([training_data, training_targets], axis = 1)

# Check for missing values
missing_values = training_data.isnull().sum()

missing_values_count = missing_values[missing_values > 0]
total_missing_values = missing_values_count.sum()

print(missing_values_count)
print("Total Missing Values:", total_missing_values)

# Checking the shapes of training data and targets
training_data_shape = training_data.shape
training_targets_shape = training_targets.shape

print("Training data Shape: ", training_data_shape)
print("Training targets Shape: ", training_targets_shape)

def knn_impute():
    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    training_data_knn_imputed = knn_imputer.fit_transform(training_data)
    training_data_knn_imputed = pd.DataFrame(training_data_knn_imputed, columns=training_data.columns)

    # Split data into training and validation sets
    X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(training_data_knn_imputed, training_targets, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier and train
    rf_classifier_knn = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier_knn.fit(X_train_knn, y_train_knn.values.ravel())

    # Predict on validation set
    y_pred_knn = rf_classifier_knn.predict(X_val_knn)

    # Calculate accuracy
    accuracy_knn_imputed = accuracy_score(y_val_knn, y_pred_knn)

    return accuracy_knn_imputed

def mean_impute():
    # Apply mean imputation
    training_data_mean_imputed = training_data.fillna(training_data.mean())

    # Split data into training and validation sets
    X_train_mean, X_val_mean, y_train_mean, y_val_mean = train_test_split(training_data_mean_imputed, training_targets, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier and train
    rf_classifier_mean = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier_mean.fit(X_train_mean, y_train_mean.values.ravel())

    # Predict on validation set
    y_pred_mean = rf_classifier_mean.predict(X_val_mean)

    # Calculate accuracy
    accuracy_mean_imputed = accuracy_score(y_val_mean, y_pred_mean)

    return accuracy_mean_imputed

def median_impute():
    # Apply median imputation
    training_data_median_imputed = training_data.fillna(training_data.median())

    # Split data into training and validation sets
    X_train_median, X_val_median, y_train_median, y_val_median = train_test_split(training_data_median_imputed, training_targets, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier and train
    rf_classifier_median = RandomForestClassifier(n_estimators=100, ccp_alpha=0, random_state=42)
    rf_classifier_median.fit(X_train_median, y_train_median.values.ravel())

    # Predict on validation set
    y_pred_median = rf_classifier_median.predict(X_val_median)

    # Calculate accuracy
    accuracy_median_imputed = accuracy_score(y_val_median, y_pred_median)

    return accuracy_median_imputed


if __name__ == "__main__":
    accuracy_knn_imputed = knn_impute()
    print("Accuracy of KNN imputed data: ", accuracy_knn_imputed)
    accuracy_mean_imputed = mean_impute()
    print("Accuracy of mean imputed data: ", accuracy_mean_imputed)
    accuracy_median_imputed = median_impute()
    print("Accuracy of median imputed data: ", accuracy_median_imputed)