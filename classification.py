import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


class classification():
     def __init__(self,clf_opt='lr',no_of_selected_features=None):
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of classifiers  
     def classification_pipeline(self):    
    # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 = DecisionTreeClassifier(max_depth=50)
#            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
            clf_parameters = {
            'clf__base_estimator':(be1,be2,be3),
            'clf__random_state':(0,10),
            }      
    # Decision Tree
        elif self.clf_opt=='dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':('auto', 'sqrt', 'log2'),
            'clf__max_depth':(10,40,45,60),
            'clf__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 
    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
    # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,100),
            }         
    # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy','gini'),       
            'clf__n_estimators':(30,50,100),
            'clf__max_depth':(10,20,30,50,100,200),
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(0.1,1,100),
            'clf__kernel':('linear','rbf','polynomial'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters    
 
# Statistics of individual classes
     def get_class_statistics(self,labels):
        class_statistics=Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))
       
# Load the data 
     def get_data(self,filename):
    # Load the file using Pandas       
        reader=pd.read_csv(filename)  
        
    # Select all rows except the ones belong to particular class'
        # mask = reader['class'] == 9
        # reader = reader[~mask]
        
        data=reader.iloc[:, :-1]
        labels=reader['target']

        self.get_class_statistics(labels)          

        return data, labels
    
# Classification using the Gold Statndard after creating it from the raw text    
     def classification(self):  
   # Get the data
        data,labels=self.get_data('data/training_data.csv')
        data=np.asarray(data)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=5)
        predicted_class_labels=[]; actual_class_labels=[]; 
        count=0; probs=[]
        for train_index, test_index in skf.split(data,labels):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count+=1                
            print('Training Phase '+str(count))
            clf,clf_parameters=self.classification_pipeline()
            pipeline = Pipeline([
                        ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
                #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                        ('clf', clf),])
            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_macro',cv=10)          
            grid.fit(X_train,y_train)     
            clf= grid.best_estimator_  
            # print('\n\n The best set of parameters of the pipiline are: ')
            # print(clf)     
            predicted=clf.predict(X_test)  
            print(predicted)
            predicted_probability = clf.predict_proba(X_test) 
            print(predicted_probability)
            for item in predicted_probability:
                probs.append(float(max(item)))
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)           
        confidence_score=statistics.mean(probs)-statistics.variance(probs)
        confidence_score=round(confidence_score, 3)
        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n') 
       
    # Evaluation
        class_names=list(Counter(labels).keys())
        class_names = [str(x) for x in class_names] 
        print('\n\n The classes are: ')
        print(class_names)      
       
        print('\n ##### Classification Report on Training Data ##### \n')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        
                
        pr=precision_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Precision:\t'+str(pr)) 
        
        rl=recall_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Recall:\t'+str(rl))
        
        fm=f1_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n F1-Score:\t'+str(fm))
        
        # Experiments on Given Test Data during Test Phase
        if confidence_score>0.85:
            print('\n ***** Classifying Test Data ***** \n')   
            predicted_cat=[]

            data,tst_cat=self.get_data('data/training_data.csv')
            tst_data=np.asarray(data)
            predicted=clf.predict(tst_data)
            print('\n ##### Classification Report ##### \n')
            print(classification_report(tst_cat, predicted, target_names=class_names)) 
        else:   
            print("\n\n ******* The classifier's condidence score ("+ str(confidence_score)+") is poor ******* \n")