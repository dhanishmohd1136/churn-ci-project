import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


def train_model(data_path="data/customer_churn.csv"):
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['Churn']).copy()

    X = df.drop(columns=["CustomerID","Churn"])

    y = df['Churn']

    numerical = X.select_dtypes(include=['number']).columns.tolist()
    categorical = X.select_dtypes(include=['object']).columns.tolist()

    numerical_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('scaler',StandardScaler())
    ])

    categorical_pipleline = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocess = ColumnTransformer([
        ('numerical',numerical_pipeline,numerical),
        ('categorical',categorical_pipleline,categorical)
    ])
    

    model = Pipeline([
        ('preprocess',preprocess),
        ('classifier',LogisticRegression(max_iter=1000,random_state=43))
    ])


    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        random_state=43,
        stratify=y,
        test_size=.2
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    metrics = {
        "accuracy":accuracy_score(y_test,pred),
        "precision":precision_score(y_test,pred),
        "f1":f1_score(y_test,pred)
    }

    return model,X_train,metrics


    