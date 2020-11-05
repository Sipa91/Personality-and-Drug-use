import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def rename_columns(df):
    #Renaming Personality trait columns
    df=df.rename(columns={"Escore":"Extraversion", "Nscore":"Neuroticism", "Oscore":"Openness", 
                      "Ascore":"Agreeableness","Cscore":"Conscientiousness", "Impulsive":"Impulsiveness",
                     "SS":"Sensation_Seeking"})

    #Renaming drug columns
    df=df.rename(columns={"Amphet":"Amphetamines", "Amyl":"Amyl_Nitrite", "Coke":"Cocaine", 
                      "Legalh":"Legal_Highs","Caff":"Caffein", "Choc":"Chocolate",
                     "Shrooms":"Mushrooms"})
    return df

def drop_colums(df, column_names):
    for i in column_names:
        df = df.drop(i, axis=1)
    return df
        

def drop_semer(df):
    #Handle the semer columns
    df = df[df['Semer'] == "CL0"]
    df = df.drop("Semer", axis=1)
    return df

def transform_countries(df):
    #converting the countries
    country = ['USA' if c < -0.5 else 
            'New Zealand' if c > -0.5 and c < -0.4 else 
            'Other' if c > -0.4 and c < -0.2 else 
            'Australia' if c > -0.2 and c < 0 else 
            'Ireland' if c > 0 and c < 0.23 else 
            'Canada' if c > 0.23 and c < 0.9 else 
            'UK' 
            for c in df['Country']]
    
    df['Country'] = country
    return df

def transform_education(df):
    education = ['Left school before 16 years' if e <-2 else 
             'Left school at 16 years' if e > -2 and e < -1.5 else 
             'Left school at 17 years' if e > -1.5 and e < -1.4 else 
             'Left school at 18 years' if e > -1.4 and e < -1 else 
             'Some college or university, no certificate or degree' if e > -1 and e < -0.5 else 
             'Professional certificate/ diploma' if e > -0.5 and e < 0 else 
             'University degree' if e > 0 and e < 0.5 else 
             'Masters degree' if e > 0.5 and e < 1.5 else 
             'Doctorate degree' 
             for e in df['Education']]

    df["Education"] = education
    return df

def transform_age(df):
    age = ['18-24' if a <= -0.9 else 
        '25-34' if a >= -0.5 and a < 0 else 
        '35-44' if a > 0 and a < 1 else 
        '45-54' if a > 1 and a < 1.5 else 
        '55-64' if a > 1.5 and a < 2 else 
        '65+' 
        for a in df['Age']]

    df["Age"] = age
    return df

def transform_gender(df):
    # women == 1 and men == 0
    df.Gender = df.Gender.apply(lambda x: 1 if x > 0 else 0)
    return df

def transform_drugs(df):
    #tranforming the drugs columns
    drugs_columns = ['Alcohol','Amphetamines','Amyl_Nitrite','Benzos','Caffein','Cannabis','Chocolate','Cocaine','Crack',
                    'Ecstasy','Heroin','Ketamine','Legal_Highs', 'LSD','Meth','Mushrooms','Nicotine','VSA']

    # Changing staments concerning drug usage into integers
    for i in drugs_columns:
        df[i] = df[i].apply(lambda x: int(x[-1]))
    return df

def drug_columns():
    drugs_columns = ['Alcohol','Amphetamines','Amyl_Nitrite','Benzos','Caffein','Cannabis','Chocolate','Cocaine','Crack',
                    'Ecstasy','Heroin','Ketamine','Legal_Highs', 'LSD','Meth','Mushrooms','Nicotine','VSA']
    return drugs_columns

def split_users(df):
    for i in drug_columns():
        df[i] = df[i].apply(lambda x: 0 if x <3 else 1)
    return df

def split_drugs(df):
    df_temp = df
    df_temp = df_temp.drop("Alcohol", axis=1)
    df_temp = df_temp.drop("Nicotine", axis=1)

    df_temp["illegal_drugs"] = df_temp.iloc[:, 13:].sum(axis=1)
    df_temp["illegal_drugs"] = df_temp["illegal_drugs"].apply(lambda x: 0 if x<1 else 1)

    df = pd.concat([df_temp["illegal_drugs"], df], axis=1)
    return df

def drop_oldies(df):
    df = df.query("Age != '65+'")
    return df

def drop_drugs(df):
    illegal_drugs = ['Amphetamines','Amyl_Nitrite','Benzos','Cannabis','Cocaine','Crack',
                    'Ecstasy','Heroin','Ketamine','Legal_Highs', 'LSD','Meth','Mushrooms','VSA']
    
    df = df.drop(illegal_drugs, axis=1)
    return df

def get_dummies(df):
    age = pd.get_dummies(df['Age'],drop_first=True)
    education = pd.get_dummies(df['Education'],drop_first=True)
    country = pd.get_dummies(df['Country'],drop_first=True)

    df = df.drop(["Age","Education", "Country"], axis=1)
    df = pd.concat([df,education,age,country],axis=1)
    return df

def balance(df):
    count_class_1, count_class_0 = df.illegal_drugs.value_counts()

    drugs_class_0 = df[df['illegal_drugs'] == 0]
    drugs_class_1 = df[df['illegal_drugs'] == 1]

    drugs_class_1_under = drugs_class_1.sample(count_class_0, replace=True)

    df = pd.concat([drugs_class_1_under, drugs_class_0], axis=0)
    return df




