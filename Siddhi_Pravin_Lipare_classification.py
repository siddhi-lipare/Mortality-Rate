import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #Logistic Regression Classifier
from sklearn.svm import SVC #Support Vector Classifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, precision_score, recall_score, confusion_matrix #Evaluation metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score #Cross Validation
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE, RFECV, SequentialFeatureSelector as SFS #Feature Selection
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier, plot_importance #XGBoost Classifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC

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
                'clf__n_estimators': [80, 90, 100, 120, 140], # No of trees in the forest
                'clf__max_depth': [None, 10, 20, 50], # Maximum depth of the tree
                'clf__min_samples_split': [2, 4, 3, 8], # Minimum number of samples required to split an internal node
                'clf__criterion': ['gini', 'entropy', 'mse'], # Function to measure the quality of a split
                'feature_selection__k': [30, 45, 47],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        elif self.clf_opt == 'lsvc':
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(random_state=42)
            param_grid = {
                'clf__penalty': ['l1', 'l2'], # Norm used in the penalization
                'clf__C': [0.1, 0.01, 0.2], # Regularization parameter
                'clf__loss': ['hinge', 'squared_hinge'], # Loss function
                'clf__max_iter': [70, 80, 100], # Maximum number of iterations
                'clf__tol': [0.0001, 0.001, 0.01], # Tolerance for stopping criteria
                'feature_selection__k': [30, 47],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }

        elif self.clf_opt == 'dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'clf__criterion': ['gini', 'entropy', 'mse'], # Function to measure the quality of a split
                'clf__max_features': ['auto', 'sqrt', 'log2'], # Number of features to consider when looking for the best split
                'clf__max_depth': [1, 5, 10, 20], # Maximum depth of the tree
                'clf__ccp_alpha': [0.009, 0.01, 0.05, 0.1], # Complexity parameter used for Minimal Cost-Complexity Pruning
                'clf__min_samples_split': [2, 4, 6], # Minimum number of samples required to split an internal node
                'feature_selection__k': [30, 45, 47],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(random_state=42)
            param_grid = {
                'clf__penalty': ['l1', 'l2'], # Norm used in the penalization
                'clf__C': [1, 0.01, 0.5, 0.02], # Regularization parameter
                'clf__class_weight': ['balanced', None], # Weights associated with classes
                'clf__max_iter': [70, 80, 100], # Maximum number of iterations 
                'clf__solver': ['liblinear', 'saga'], # Algorithm to use in the optimization problem
                'clf__tol': [0.0001, 0.001, 0.01], # Tolerance for stopping criteria
                # 'feature_selection__k': [30, 47],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        elif self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = RandomForestClassifier(max_depth=20, n_estimators=50, min_samples_split=3,min_samples_leaf=1,criterion='gini')
            be2 = DecisionTreeClassifier(max_depth=40,criterion='entropy',max_features='log2')
            be3=  LogisticRegression(class_weight='balanced',fit_intercept='True', C=1, max_iter=100,penalty='l1',solver='liblinear')
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            param_grid = {
                'clf__base_estimator': [be1, be2, be3],
                'clf__n_estimators': [50, 120, 130],
                'clf__learning_rate': [0.01, 0.1, 1],
                'clf__random_state': [0, 10],
                # 'feature_selection__k': [30, 45, 47],
            }
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = SVC(random_state=42)
            param_grid = {
                'clf__C': [0.1, 0.6, 0.5], # Regularization parameter
                'clf__kernel': ['poly', 'rbf', 'sigmoid'], # Kernel type
                'clf__degree': [1, 2, 3, 4], # Degree of the polynomial kernel function
                'clf__gamma': ['scale', 'auto'], # Kernel coefficient
                'feature_selection__k': [30, 45],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        elif self.clf_opt == 'knn':
            print('\n\t### Training KNN Classifier ### \n')
            clf = KNeighborsClassifier()
            param_grid = {
                'clf__n_neighbors': [5, 10, 15, 20], # Number of neighbors
                'clf__weights': ['uniform', 'distance'], # Weight function used in prediction
                'clf__metric': ['euclidean', 'manhattan'], # Distance metric
                # 'feature_selection__k': [30, 45],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        elif self.clf_opt == 'xg':
            print('\n\t### Training XGBoost Classifier ### \n')
            clf = XGBClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [80, 90, 100], # No of trees in the forest
                'clf__max_depth': [3, 5, 7], # Maximum depth of the tree
                'clf__learning_rate': [0.01, 0.1, 0.2], # Boosting learning rate
                'clf__subsample': [0.8, 0.9, 1.0], # Subsample ratio of the training instances
                'clf__colsample_bytree': [0.8, 0.9, 1.0], # Subsample ratio of columns when constructing each tree
                'clf__scale_pos_weight': [6.34], # Control the balance of positive and negative weights. 6.34 was chosen since a typical value to consider: sum(negative instances) / sum(positive instances)
                'feature_selection__k': [30, 40],  # Add k to the parameter grid. Uncomment this line if using feature selection using SelectKBest
            }
        else:
            print('Invalid classifier option')
            return None, None

        imputer = self.get_imputer() # Get the imputer
        return clf, imputer, param_grid # Return the classifier, imputer and parameter grid


    def get_imputer(self):
        if self.impute_opt == 'mean': 
            return SimpleImputer(strategy='mean') # Impute missing values using the mean along each column
        elif self.impute_opt == 'median':
            return SimpleImputer(strategy='median') # Impute missing values using the median along each column
        elif self.impute_opt == 'mode':
            return SimpleImputer(strategy='most_frequent') # Impute missing values using the mode along each column
        elif self.impute_opt == 'knn':
            return KNNImputer(n_neighbors=3) # Impute missing values using KNN imputation method
        else:
            print('Invalid imputation option')
            return None

    def classification(self):
        X = pd.read_csv("data/training_data.csv")
        Y = pd.read_csv("data/training_data_targets.csv", header=None)
        # X_testdata = pd.read_csv("data/test_data.csv")

        # Imputation
        imputer = self.get_imputer()
        if imputer is None:
            return

        X_imputed = pd.DataFrame(data=imputer.fit_transform(X), columns=X.columns)

        # Tried various imputation methods but StandardScaling gave best results.
        # Standard Scaling after imputation
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(data=scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # X_testdata_scaled = pd.DataFrame(data=scaler.transform(X_testdata), columns=X_testdata.columns)

        # # Robust Scaling after imputation
        # scaler = RobustScaler()
        # X_scaled = pd.DataFrame(data=scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # X_testdata_scaled = pd.DataFrame(data=scaler.transform(X_testdata), columns=X_testdata.columns)

        # # MinMax Scaling after imputation
        # scaler = MinMaxScaler()
        # X_scaled = pd.DataFrame(data=scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # X_testdata_scaled = pd.DataFrame(data=scaler.transform(X_testdata), columns=X_testdata.columns)

        # # MaxAbs Scaling after imputation
        # scaler = MaxAbsScaler()
        # X_scaled = pd.DataFrame(data=scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # X_testdata_scaled = pd.DataFrame(data=scaler.transform(X_testdata), columns=X_testdata.columns)

        # Drop 'age' feature
        X_scaled.drop(columns = ['age'],inplace=True)
        # X_testdata.drop('age', axis=1, inplace=True)

        # Specify categorical columns
        categorical_cols = ['gendera', 'hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes',
                                'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD']

        # Used OneHotEncoder Function for one-hot encoding
        one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
        X_encoded = pd.DataFrame()
        for col in categorical_cols:
            if X_scaled[col].nunique() == 2:
                encoded_col = one_hot_encoder.fit_transform(X_scaled[[col]])
                encoded_col_df = pd.DataFrame(encoded_col, columns=[f"{col}_{int(i)}" for i in range(encoded_col.shape[1])])
                X_encoded = pd.concat([X_encoded, encoded_col_df], axis=1)
        # X_testdata_encoded = pd.DataFrame(encoder.transform(X_testdata[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

        # Combine encoded features with the rest of the data
        X_scaled = pd.concat([X_scaled, X_encoded], axis=1)

        # Drop the original categorical columns
        X_scaled = X_scaled.drop(columns=categorical_cols)

        # Use train_test_split with stratify after imputation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42, stratify=Y)

        # # Tried various oversampling methods but SMOTE gave the best results
        # SMOTE oversampling to balance classes
        oversampler = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

        # # Random oversampling to balance classes
        # oversampler = RandomOverSampler(random_state=42)
        # X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

        # # ADASYN oversampling to balance classes
        # oversampler = ADASYN(random_state=42)
        # X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

        # # Manual oversampling to balance classes
        # X_train_balanced = pd.concat([X_train, X_train[y_train[0] == 1].sample(n=1000, replace=True, random_state=42)])
        # y_train_balanced = pd.concat([y_train, y_train[y_train[0] == 1].sample(n=1000, replace=True, random_state=42)])


        # Classification
        clf, _, param_grid = self.classification_pipeline()
        if clf is None:
            return

        # Tried various Feature Selection methods but chose SelectKBest with ANOVA F1 Score as it gave the best results

        # # Add feature selection to the pipeline - SelectKBest with ANOVA F-value
        # feature_selection = SelectKBest(score_func=f_classif, k=self.no_of_selected_features)
        # pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        # pipeline = Pipeline(pipeline_steps)

        # Add feature selection to the pipeline - Mutual Information
        feature_selection = SelectKBest(score_func=mutual_info_classif, k=45)
        pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        pipeline = Pipeline(pipeline_steps)

        # # Add feature selection to the pipeline - Chi-Squared
        # feature_selection = SelectKBest(score_func=chi2, k='all')
        # pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        # pipeline = Pipeline(pipeline_steps)

        # # Add feature selection to the pipeline - Recursive Feature Elimination
        # corr_matrix = X_scaled.corr().abs()
        # upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        # X_scaled_selected = X_scaled.drop(columns=to_drop)
        # print("### Shape of Data After Feature Selection ###")
        # print()
        # print("Shape of Training Data", X_scaled_selected.shape)
        # feature_selection = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=40)
        # pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        # pipeline = Pipeline(pipeline_steps)

        # # Add feature selection to the pipeline - Recursive Feature Elimination with Cross Validation
        # feature_selection = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
        # pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        # pipeline = Pipeline(pipeline_steps)

        # # Add feature selection to the pipeline - Sequential Feature Selector with Cross Validation
        # feature_selection = SFS(estimator=clf, k_features=self.no_of_selected_features, forward=True, scoring='accuracy', cv=5)
        # pipeline_steps = [('imputer', imputer), ('feature_selection', feature_selection), ('clf', clf)]
        # pipeline = Pipeline(pipeline_steps)


        scorer = make_scorer(f1_score, average='weighted')
        # scorer = make_scorer(f1_score, average='micro')
        # scorer = make_scorer(f1_score, average='macro')

        # Fit the pipeline using GridSearchCV
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=5)
        grid_search.fit(X_train_balanced, y_train_balanced.values.ravel())

        # Print the best hyperparameters
        print(f"\nBest Hyperparameters: {grid_search.best_params_}")

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        # # Evaluate the best model
        # print("\nF1 Score: ", f1_score(y_test, y_pred, average='weighted'))
        # print("Accuracy: ", accuracy_score(y_test, y_pred))
        # print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
        # print("Recall: ", recall_score(y_test, y_pred, average='weighted'))

        # print("\nF1 Score: ", f1_score(y_test, y_pred, average='micro'))
        # print("Accuracy: ", accuracy_score(y_test, y_pred))
        # print("Precision: ", precision_score(y_test, y_pred, average='micro'))
        # print("Recall: ", recall_score(y_test, y_pred, average='micro'))

        # print("\nF1 Score: ", f1_score(y_test, y_pred, average='macro'))
        # print("Accuracy: ", accuracy_score(y_test, y_pred))
        # print("Precision: ", precision_score(y_test, y_pred, average='macro'))
        # print("Recall: ", recall_score(y_test, y_pred, average='macro'))

        # Classification Report
        clf_report = classification_report(y_test, y_pred)
        print(clf_report)


        # Display Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # # Make predictions on the test data
        # y_pred_testdata = best_model.predict(X_testdata_scaled)
        # y_pred_testdata = pd.DataFrame(y_pred_testdata)
        # y_pred_testdata.to_csv('data/test_data_predictions.csv', index=False)


# AdaBoost Classifier is the best classifier for this dataset since it has the highest F1 Score and Accuracy as compared to other classifiers
# Apart from AdaBoost Classifier, XGBoost Classifier and Logistic Regression also gave good results
