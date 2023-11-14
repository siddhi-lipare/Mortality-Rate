import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

# Performed oversampling using the Synthetic Minority Over-sampling Technique (SMOTE)
from imblearn.over_sampling import SMOTE

class Classification:
    def __init__(self, clf_opt='rf', impute_opt='mean', no_of_selected_features=None):
        self.clf_opt = clf_opt
        self.impute_opt = impute_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features is not None:
            self.no_of_selected_features = int(self.no_of_selected_features)

    def classification_pipeline(self):
        if self.clf_opt == 'rf':
            clf = RandomForestClassifier(random_state=42)
        elif self.clf_opt == 'dt':
            clf = DecisionTreeClassifier(random_state=42)
        elif self.clf_opt == 'ab':
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=42), random_state=42)
        else:
            print('Invalid classifier option')
            return None, None

        if self.impute_opt == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif self.impute_opt == 'median':
            imputer = SimpleImputer(strategy='median')
        elif self.impute_opt == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            print('Invalid imputation option')
            return None, None

        return clf, imputer

    def oversample_smote(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def mean_imputation(self, df):
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(data=imputer.fit_transform(df), columns=df.columns)
        return X_imputed

    def median_imputation(self, df):
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(data=imputer.fit_transform(df), columns=df.columns)
        return X_imputed

    def knn_imputation(self, df):
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(data=imputer.fit_transform(df), columns=df.columns)
        return X_imputed

    def classification(self):
        # Load data
        X = pd.read_csv("data/training_data.csv")
        Y = pd.read_csv("data/training_data_targets.csv", header=None)

        # Imputation
        if self.impute_opt == 'mean':
            X_imputed = self.mean_imputation(X)
        elif self.impute_opt == 'median':
            X_imputed = self.median_imputation(X)
        elif self.impute_opt == 'knn':
            X_imputed = self.knn_imputation(X)
        else:
            print('Invalid imputation option')
            return

        # Oversample
        X_resampled, y_resampled = self.oversample_smote(X_imputed, Y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        # Classification
        clf, imputer = self.classification_pipeline()
        if clf is None or imputer is None:
            return

        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_sc = f1_score(y_test, y_pred)

        print(f'F1-Score with {self.impute_opt} imputation and {self.clf_opt} classifier: {f1_sc:.2f}')
        print(f'Accuracy with {self.impute_opt} imputation and {self.clf_opt} classifier: {accuracy:.2f}')


