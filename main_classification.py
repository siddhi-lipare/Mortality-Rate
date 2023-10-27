from classification import classification

import warnings
warnings.filterwarnings("ignore")


clf=classification(clf_opt='dt',
                        no_of_selected_features=4)

clf.classification()