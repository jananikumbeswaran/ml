from fastapi import FastAPI, UploadFile, File
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd

app = FastAPI()

models = {}
trained = False

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global models, trained
    data_df = pd.read_csv(file.file)
    x = data_df.drop('target', axis=1)
    y = data_df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }
    accuracy_results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if name == "Linear Regression":
            y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[name] = f"{accuracy * 100:.2f}%"

    trained = True
    return {"status": "Training completed", "accuracies": accuracy_results}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not trained:
        return { "Model not trained yet train first."}

    test_df = pd.read_csv(file.file)
    x_test_data = test_df.drop('target', axis=1)
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(x_test_data)
        predictions[name] = y_pred.tolist()

    return {"predictions": predictions}