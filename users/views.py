from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load and preprocess data


from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from django.conf import settings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# =========================================================
# TRAINING VIEW
# =========================================================
def training(request):

    dataset_path = os.path.join(settings.MEDIA_ROOT, 'Darpa.csv')
    dataset = pd.read_csv(dataset_path)

    # -----------------------------
    # ENCODE STRING FEATURE COLUMNS
    # -----------------------------
    feature_encoders = {}

    for col in dataset.columns:
        if dataset[col].dtype == 'object' and col != "attack_cat":
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
            feature_encoders[col] = le

    # -----------------------------
    # ENCODE TARGET COLUMN
    # -----------------------------
    attack_encoder = LabelEncoder()
    dataset['attack_cat'] = attack_encoder.fit_transform(dataset['attack_cat'])

    dataset.fillna(dataset.mean(), inplace=True)

    # -----------------------------
    # SPLIT FEATURES & TARGET
    # -----------------------------
    y = dataset['attack_cat']
    X = dataset.drop(columns=['attack_cat'])

    feature_names = X.columns.tolist()

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = {}

    # -----------------------------
    # TRAIN FUNCTION
    # -----------------------------
    def train_model(name, model):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results[name] = acc

        joblib.dump({
            "model": model,
            "features": feature_names,
            "scaler": scaler,
            "feature_encoders": feature_encoders,
            "attack_encoder": attack_encoder
        }, os.path.join(settings.MEDIA_ROOT, f'{name}_model.pkl'))


    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    train_model("RandomForest", RandomForestClassifier())
    train_model("AdaBoost", AdaBoostClassifier())
    train_model("LogisticRegression", LogisticRegression(max_iter=500))
    train_model("KNN", KNeighborsClassifier())
    train_model("SVM", svm.SVC())
    train_model("DecisionTree", DecisionTreeClassifier())
    train_model("MLP", MLPClassifier(max_iter=500))
    train_model("XGBoost", XGBClassifier())


    # -----------------------------
    # PLOT GRAPH
    # -----------------------------
    plt.figure(figsize=(10,5))
    plt.bar(results.keys(), results.values())
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Cyber Attack Detection Model Comparison")

    graph_path = os.path.join(settings.MEDIA_ROOT, 'accuracy_graph.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    results_df = pd.DataFrame(list(results.items()), columns=['Algorithm','Accuracy'])

    return render(request, 'users/accuracy.html', {
        'results': results_df.to_html(index=False),
        'graph_url': '/media/accuracy_graph.png'
    })


# =========================================================
# PREDICTION VIEW
# =========================================================
def prediction(request):

    context = {}

    model_path = os.path.join(settings.MEDIA_ROOT, "RandomForest_model.pkl")

    if not os.path.exists(model_path):
        context["error"] = "Model not trained yet. Please run training first."
        return render(request, "users/prediction.html", context)

    model_data = joblib.load(model_path)

    model = model_data["model"]
    train_features = model_data["features"]
    scaler = model_data["scaler"]
    feature_encoders = model_data.get("feature_encoders", {})
    attack_encoder = model_data.get("attack_encoder")

    if request.method == "POST":

        try:
            csv_file = request.FILES["csv_file"]
            df = pd.read_csv(csv_file)

            # -----------------------------
            # ENCODE FEATURE COLUMNS
            # -----------------------------
            for col, le in feature_encoders.items():
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )

            # -----------------------------
            # ALIGN COLUMNS
            # -----------------------------
            for col in train_features:
                if col not in df.columns:
                    df[col] = 0

            df = df[train_features]

            # -----------------------------
            # SCALE
            # -----------------------------
            df_scaled = scaler.transform(df)

            # -----------------------------
            # PREDICT
            # -----------------------------
            predictions = model.predict(df_scaled)

            # CONVERT NUMERIC TO ATTACK NAME
            if attack_encoder is not None:
                predictions = attack_encoder.inverse_transform(predictions)

            df["Predicted_Attack"] = predictions

            context["table"] = df.to_html(classes="table table-bordered table-striped")

        except Exception as e:
            context["error"] = f"Prediction failed: {str(e)}"

    return render(request, "users/prediction.html", context)
import os

def ViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'Darpa.csv')
    import pandas as pd
    df = pd.read_csv(dataset, nrows=100)
    df = df.to_html(index=None)
    return render(request, 'users/viewData.html', {'data': df})


from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")
