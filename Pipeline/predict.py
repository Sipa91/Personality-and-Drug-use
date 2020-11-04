import sys
import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import rename_columns, drop_semer, transform_drugs, split_users, split_drugs, drop_colums, transform_countries, transform_education, transform_age, drop_oldies, transform_gender, drop_drugs, get_dummies, balance

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

#
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

print("Feature engineering")
X_test = transform_countries(X_test)
X_test = transform_education(X_test)
X_test = transform_age(X_test)
X_test = transform_gender(X_test)
X_test = drop_colums(X_test, ["Alcohol", "Nicotine", "ID", "Ethnicity"])
X_test = drop_drugs(X_test)
X_test = get_dummies(X_test)


rf_predictions_test = loaded_model.predict(X_test)

print("accuracy Test: ", accuracy_score(y_test, rf_predictions_test))

print(classification_report(y_test, rf_predictions_test))
