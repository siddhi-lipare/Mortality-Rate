from classification2 import Classification

import warnings
warnings.filterwarnings("ignore")

classifier = Classification(clf_opt='lr', impute_opt='knn')
classifier.classification()
