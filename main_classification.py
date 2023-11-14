from classification import Classification

import warnings
warnings.filterwarnings("ignore")

classifier = Classification(clf_opt='ab', impute_opt='knn', no_of_selected_features=1000)
classifier.classification()