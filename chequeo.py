import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce' )
print(df['TotalCharges'].isnull().sum())
df.dropna(subset= ['TotalCharges'], inplace=True)
print(df['TotalCharges'].isnull().sum())

df.drop(['customerID'], axis=1, inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn'] #respuesta correcta

def preprosesamiento(X, y, classification = None):
    
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    num_cols = X.select_dtypes(exclude="object").columns
    cat_cols = X.select_dtypes(include="object").columns
    
    preprocesor = ColumnTransformer(
        [
            ('num', StandardScaler(), num_cols), ('cat',OneHotEncoder(), cat_cols)
        ]
    )
    
    xtrain = preprocesor.fit_transform(xtrain)
    xtest = preprocesor.transform(xtest)
    
    # if dump:
    #     dumpfolder(
    #         preprocesor, type='preprocessor', filename='preprocessor.pkl'
    #     )
        
    
    if classification:
        le = LabelEncoder()
        ytrain = le.fit_transform(ytrain)
        ytest = le.transform(ytest)
        
        print("Distribución original en ytrain:")
        print(pd.Series(ytrain).value_counts())
        
    #     if dump:
    #         dumpfolder(le, type="preprocessor", filename="label_encoder.pkl")
    
    # elif regression:
    #     pass
    
    
    # if self.oversample:
    #     smote = SMOTE(random_state=42, sampling_strategy="minority")
    #     xtrain, ytrain = smote.fit_resample(xtrain,ytrain)
    # elif self.undersample:
    #     rus = RandomUnderSampler(random_state=42)
    #     xtrain, ytrain = rus.fit_resample(xtrain, ytrain)
    
    return xtrain, xtest, ytrain, ytest

xtrain, xtest, ytrain, ytest = preprosesamiento(X, y, classification=True)
print("Distribución en ytest:")
print(pd.Series(ytest).value_counts())

print(X.dtypes)
print(X.columns)
print(df.describe())
