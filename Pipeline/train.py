import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, make_scorer, fbeta_score, plot_confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import rename_columns, drop_semer, transform_drugs, split_users, split_drugs, drop_colums, transform_countries, transform_education, transform_age, drop_oldies, transform_gender, drop_drugs, get_dummies, balance

#import the dataset
drugs = pd.read_excel("/Users/kevintomas/Desktop/nf-sep-20/Personality-and-Drug-use/drug_consumption.xls")

# general preprocessing
drugs = rename_columns(drugs)
drugs = drop_semer(drugs)
drugs = transform_drugs(drugs)
#print(drugs.columns)
drugs = split_users(drugs)
drugs = drop_colums(drugs, ["Chocolate", "Caffein"])
drugs = drop_oldies(drugs)
drugs = split_drugs(drugs)
#print(drugs.illegal_drugs.value_counts())
drugs = balance(drugs)

#print("class 1:", drugs[drugs['illegal_drugs'] == 1].shape)
#print("class 0:", drugs[drugs['illegal_drugs'] == 0].shape)


# bis hierhin funzt es 
'''['illegal_drugs', 'ID', 'Age', 'Gender', 'Education', 'Country',
       'Ethnicity', 'Neuroticism', 'Extraversion', 'Openness', 'Agreeableness',
       'Conscientiousness', 'Impulsiveness', 'Sensation_Seeking', 'Alcohol',
       'Amphetamines', 'Amyl_Nitrite', 'Benzos', 'Caffein', 'Cannabis',
       'Chocolate', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
       'Legal_Highs', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']'''

#Splitting data
y = np.array(drugs.pop('illegal_drugs'))
X = drugs.iloc[:, 0:]

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
#in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

print("Feature engineering on train")
#X_train = drop_colums(drugs, ["Chocolate", "Caffein"])
X_train = transform_countries(X_train)
X_train = transform_education(X_train)
X_train = transform_age(X_train)
X_train = transform_gender(X_train)
X_train = drop_colums(X_train, ["Alcohol", "Nicotine", "ID", "Ethnicity"])

''' ['Age', 'Gender', 'Education', 'Country', 'Neuroticism', 'Extraversion',
       'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness',
       'Sensation_Seeking', 'Amphetamines', 'Amyl_Nitrite', 'Benzos',
       'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
       'Legal_Highs', 'LSD', 'Meth', 'Mushrooms', 'VSA']
'''
X_train = drop_drugs(X_train)

'''['Age', 'Gender', 'Education', 'Country', 'Neuroticism', 'Extraversion',
       'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness',
       'Sensation_Seeking']'''

X_train = get_dummies(X_train)



# model
print("Training a random forest model")
# Create the model with 100 trees
rf = RandomForestClassifier(n_estimators=100, 
                               random_state=50, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

rf = rf.fit(X_train, y_train)
rf_predictions_train = rf.predict(X_train)
print("accuracy Train: ", accuracy_score(y_train, rf_predictions_train))



#feature eng on test data
print("Feature engineering on test")
X_test = transform_countries(X_test)
X_test = transform_education(X_test)
X_test = transform_age(X_test)
X_test = transform_gender(X_test)
X_test = drop_colums(X_test, ["Alcohol", "Nicotine", "ID", "Ethnicity"])
X_test = drop_drugs(X_test)
X_test = get_dummies(X_test)

rf_predictions_test = rf.predict(X_test)
print("accuracy Test: ", accuracy_score(y_test, rf_predictions_test))

print(classification_report(y_test, rf_predictions_test))

print("Saving model in the model folder")
filename = 'models/rf_model.sav'
pickle.dump(rf, open(filename, 'wb'))

