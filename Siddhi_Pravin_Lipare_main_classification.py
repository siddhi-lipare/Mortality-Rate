from Siddhi_Pravin_Lipare_classification import Classification

import warnings
warnings.filterwarnings("ignore")

classifier = Classification(clf_opt='xg', impute_opt='knn')
classifier.classification()
