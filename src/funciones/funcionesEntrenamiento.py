import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


def cargar_datos_entrenamiento():
    df = pd.read_csv("../../data/ds_tradicional.csv")
    # Separar características y etiquetas
    X = df.drop(columns=["class"])  
    y = df["class"]
    return X, y

def cargar_datos_prueba():
    df = pd.read_csv("../../data/ds_tradicional_TEST.csv")
    # Separar características y etiquetas
    X = df.drop(columns=["class"])  
    y = df["class"]
    return X, y

def imprimir_forma(df):
    return df.shape, df.head(5)

def division_train_val(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=22,
        stratify=y
    )
    return X_train, X_val, y_train, y_val

def reporte_clasificacion(X, y, modelo, lineal=False):
    if not lineal:
        y_pred = modelo.predict(X)
        y_res = modelo.predict_proba(X)[:, 1]
        reporte = classification_report(y, y_pred)
    else:
        y_pred = modelo.predict(X)
        y_res = modelo.decision_function(X)
        reporte = classification_report(y, y_pred)
    
    return y_pred, y_res, reporte

def crear_matriz_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return cm, disp

def calcular_roc_auc(y, y_res):
    fpr, tpr, thresholds = roc_curve(y, y_res)
    auc_score = roc_auc_score(y, y_res)
    return fpr, tpr, thresholds, auc_score

def metricas_tpr_fpr(cm):
    TN, FP, FN, TP = cm.ravel()
    FPR = 0.0 if FP + TN == 0.0 else FP / (FP + TN)
    TPR = 0.0 if TP + FN == 0.0 else TP / (TP + FN)
    return TPR, FPR

def hacer_pepinillo(modelo, nombre, test=False):
    if test:
        os.makedirs(os.path.dirname(nombre), exist_ok=True)
        with open(nombre, "wb") as f:
            pickle.dump(modelo, f)
    else:
        with open(f"../../models/{nombre}", "wb") as f:
            pickle.dump(modelo, f)
