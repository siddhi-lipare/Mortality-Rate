from Siddhi_Pravin_Lipare_classification import Classification

import warnings
warnings.filterwarnings("ignore")

classifier = Classification(clf_opt='ab', impute_opt='knn')
classifier.classification()
