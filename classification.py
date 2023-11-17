import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

class Classification:
    def __init__(self, clf_opt='rf', impute_opt='mean', no_of_selected_features=None):
        self.clf_opt = clf_opt
        self.impute_opt = impute_opt
        self.no_of_selected_features = int(no_of_selected_features) if no_of_selected_features is not None else None

    def classification_pipeline(self):
        if self.clf_opt == 'rf':
            print('\n\t### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [50, 70, 80],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 4, 6],
                'clf__min_samples_leaf': [2, 3, 4],
                # 'clf__bootstrap': [True, False],
                'clf__criterion': ['gini', 'entropy'],
                # 'clf__class_weight': ['balanced', 'balanced_subsample', None]
            }
        elif self.clf_opt == 'dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_features': ['auto', 'sqrt', 'log2'],
                'clf__max_depth': [10, 40, 45, 60],
                'clf__ccp_alpha': [0.009, 0.01, 0.05, 0.1],
            }
        elif self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = RandomForestClassifier(max_depth=50, n_estimators=100)
            be2 = LogisticRegression(solver='liblinear', class_weight='balanced')
            be3 = DecisionTreeClassifier(max_depth=50)
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            param_grid = {
                'clf__base_estimator': [be1, be2, be3],
                'clf__n_estimators': [50, 100, 150],
                'clf__learning_rate': [0.01, 0.1, 1],
                'clf__random_state': [0, 10]
            }
        elif self.clf_opt == 'mlp':
            print('\n\t### Training MLP Classifier ### \n')
            clf = MLPClassifier(random_state=42)
            param_grid = {
                'clf__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,50,50)],
                'clf__activation': ['tanh', 'relu'],
                'clf__solver': ['sgd', 'adam'],
                'clf__alpha': [0.0001, 0.05],
                # 'clf__learning_rate': ['constant', 'adaptive']
            }
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = SVC(random_state=42)
            param_grid = {
                'clf__C': [0.1, 1, 10, 100],
                'clf__gamma': [1, 0.1, 0.01, 0.001],
                'clf__kernel': ['rbf', 'poly', 'sigmoid']
            }
        elif self.clf_opt == 'knn':
            print('\n\t### Training KNN Classifier ### \n')
            clf = KNeighborsClassifier()
            param_grid = {
                'clf__n_neighbors': [5, 10, 15, 20],
                'clf__weights': ['uniform', 'distance'],
                'clf__metric': ['euclidean', 'manhattan']
            }
        elif self.clf_opt == 'xg':
            print('\n\t### Training XGBoost Classifier ### \n')
            clf = XGBClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [80, 90, 100],
                'clf__max_depth': [3, 5, 7],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__subsample': [0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.8, 0.9, 1.0],
                'clf__scale_pos_weight': [6.34]
            }
        else:
            print('Invalid classifier option')
            return None, None

        imputer = self.get_imputer()
        return clf, imputer, param_grid


    def get_imputer(self):
        if self.impute_opt == 'mean':
            return SimpleImputer(strategy='mean')
        elif self.impute_opt == 'median':
            return SimpleImputer(strategy='median')
        elif self.impute_opt == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif self.impute_opt == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            print('Invalid imputation option')
            return None

    def oversample_smote(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def classification(self):
        # Load data
        X = pd.read_csv("data/training_data.csv")
        Y = pd.read_csv("data/training_data_targets.csv", header=None)
        # X_testdata = pd.read_csv("data/test_data.csv")

        # Imputation
        imputer = self.get_imputer()
        if imputer is None:
            return

        X_imputed = pd.DataFrame(data=imputer.fit_transform(X), columns=X.columns)

        # Standard Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(data=scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # X_test_scaled = pd.DataFrame(data=scaler.fit_transform(X_testdata), columns=X_testdata.columns)

        # Oversample
        X_resampled, y_resampled = self.oversample_smote(X_scaled, Y)

        # Classification
        clf, _, param_grid = self.classification_pipeline()
        if clf is None:
            return

        pipeline_steps = [('imputer', imputer), ('scaler', scaler), ('clf', clf)]

        pipeline = Pipeline(pipeline_steps)

        scorer = make_scorer(f1_score, average='weighted')
        
        # Use StratifiedKFold with k=5
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(X_resampled, y_resampled)):
            X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

            # Fit the pipeline using GridSearchCV
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=5)
            grid_search.fit(X_train, y_train.values.ravel())

            # Print the best hyperparameters for each fold
            print(f"\nBest Hyperparameters - Fold {fold + 1}: {grid_search.best_params_}")

            # Get the best model
            best_model = grid_search.best_estimator_

            # Make predictions on the test set using the best model
            y_pred = best_model.predict(X_test)

            # Evaluate the best model for each fold
            f1_score_final = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision_final = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_mat = confusion_matrix(y_test, y_pred)
            print(f'F1-Score of the Best Model - Fold {fold + 1}: {f1_score_final:.2f}')
            print(f'Precison of the model - Fold {fold + 1}: {precision_final:.2f}')
            print(f'Accuracy of the Best Model - Fold {fold + 1}: {accuracy:.2f}')
            print(f'Recall of the Best Model - Fold {fold + 1}: {recall:.2f}')
            print(f'Confusion Matrix of the Best Model - Fold {fold + 1}: {conf_mat}')

            # # Make predictions on the test data using the best model
            # y_test_pred = best_model.predict(X_test_scaled)

            # # Create a DataFrame with the predicted labels for each fold
            # df_predictions = pd.DataFrame(y_test_pred)

            # # Save the DataFrame to a CSV file for each fold
            # df_predictions.to_csv(f'predicted_labels_fold_{fold + 1}.csv', index=False)
