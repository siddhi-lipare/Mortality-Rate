# Mortality-Rate
Prediction of Mortality Rate of Heart Failure Patients Admitted to ICU

## Objective
- The objective of this project is to predict the factors of hospital mortality for the patients who admitted to ICUs (Intensive Care Units) due to heart failure. 
- The data is collected from MIMIC-III database (version 1.4, 2016), which is a publicly available critical care database containing de-identified data on 46,520 patients and 58,976 admissions to the ICU of the Beth Israel Deaconess Medical Center, Boston, USA, between 1 June, 2001 and 31 October, 2012.

----------
*Prerequisites*
-------------
- `Python 3.10` 
- Install necessary packages using `pip install -r requirements.txt`

----------
*Get the output:*
----------
- Run the Following command:
```bash
python main_classification.py
```
----------
*Results and Findings:*
----------
- I achieved a maximum F1-Score of 0.95 and an accuracy of 95% upon training the model on the Random Forest Classifier and AdaBoost Classifier imputing the missing values using knn imputation method. 
