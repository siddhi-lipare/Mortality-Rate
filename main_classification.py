from classification import Classification

import warnings
warnings.filterwarnings("ignore")

classifier = Classification(clf_opt='dt', impute_opt='mean', no_of_selected_features=1000)
classifier.classification()