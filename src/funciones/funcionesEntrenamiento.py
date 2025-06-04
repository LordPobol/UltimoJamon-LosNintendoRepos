# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de la fase de entrenamiento de modelos tradicionales.
# 
# --------------------------------------------------

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
    """
    Descripción: Carga el conjunto de datos de entrenamiento desde un archivo CSV, separando todos los atributos (X) de las etiquetas (y).
    Entrada: Ninguna.
    Salida - X (DataFrame): Atributos del conjunto de entrenamiento.
           - y (Series): Etiquetas del conjunto de entrenamiento.
    """
    df = pd.read_csv("../../data/ds_tradicional.csv")
    # Separar características y etiquetas
    X = df.drop(columns=["class"])  
    y = df["class"]
    return X, y

def cargar_datos_prueba():
    """
    Descripción: Carga el conjunto de datos de prueba desde un archivo CSV, separando todos los atributos (X) de las etiquetas (y).
    Entrada: Ninguna.
    Salida - X (DataFrame): Características del conjunto de prueba.
           - y (Series): Etiquetas del conjunto de prueba.
    """
    df = pd.read_csv("../../data/ds_tradicional_TEST.csv")
    # Separar características y etiquetas
    X = df.drop(columns=["class"])  
    y = df["class"]
    return X, y

def imprimir_forma(df):
    """
    Descripción: Retorna las dimensiones del DataFrame y las primeras 5 filas para una inspección rápida.
    Entrada - df (DataFrame): Conjunto de datos a inspeccionar.
    Salida - shape (tuple): Dimensiones del DataFrame.
           - head (DataFrame): Primeras 5 filas del DataFrame.
    """
    return df.shape, df.head(5)

def division_train_val(X, y):
    """
    Descripción: Divide el conjunto de datos en entrenamiento (80%) y validación (20%), manteniendo la proporción de clases.
    Entrada - X (DataFrame): Conjunto de atributos.
            - y (Series): Conjunto de las etiquetas clasificatorias.
    Salida - X_train, X_val, y_train, y_val: Subconjuntos para entrenamiento y validación.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=22,
        stratify=y
    )
    return X_train, X_val, y_train, y_val

def reporte_clasificacion(X, y, modelo, lineal=False):
    """
    Descripción: Generar predicciones, calcular probabilidades de predicción o funciones de decisión, y crea un reporte de clasificación.
    Entrada - X (DataFrame): Conjunto de atributos.
            - y (Series): Conjunto de las etiquetas verdaderas.
            - modelo (Model): Modelo entrenado.
            - lineal (bool): Indica si el modelo es lineal, en caso de que use una función de decisión en vez de una probabilidad de predicción.
    Salida - y_pred (array): Etiquetas predichas.
           - y_res (array): Probabilidades o puntuaciones de decisión.
           - reporte (str): Métricas de desempeño del modelo.
    """
    y_pred = modelo.predict(X)
    reporte = classification_report(y, y_pred)
    if not lineal:
        y_res = modelo.predict_proba(X)[:, 1]
    else:
        y_res = modelo.decision_function(X)
    return y_pred, y_res, reporte

def crear_matriz_confusion(y_test, y_pred):
    """
    Descripción: Calcula y prepara los valores  de la matriz de confusión para su visualización.
    Entrada - y_test (array): Etiquetas verdaderas.
            - y_pred (array): Etiquetas predichas.
    Salida - cm (array): Matriz de confusión en forma de matriz.
           - disp (objeto): Objeto para mostrar la matriz de confusión.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return cm, disp

def calcular_roc_auc(y, y_res):
    """
    Descripción: Calcula la curva ROC y el valor AUC del modelo.
    Entrada - y (array): Etiquetas verdaderas.
            - y_res (array): Probabilidades o puntuaciones del modelo.
    Salida - fpr (array): Tasas de falsos positivos.
           - tpr (array): Tasas de verdaderos positivos.
           - thresholds (array): Umbrales de decisión.
           - auc_score (float): Área bajo la curva ROC.
    """
    fpr, tpr, thresholds = roc_curve(y, y_res)
    auc_score = roc_auc_score(y, y_res)
    return fpr, tpr, thresholds, auc_score

def metricas_tpr_fpr(cm):
    """
    Descripción: Calcula las tasas de verdaderos positivos (TPR) y falsos positivos (FPR) a partir de una matriz de confusión.
    Entrada - cm (array): Matriz de confusión en forma de matriz.
    Salida - TPR (float): Tasa de verdaderos positivos.
             FPR (float): Tasa de falsos positivos.
    """
    TN, FP, FN, TP = cm.ravel()
    FPR = 0.0 if FP + TN == 0.0 else FP / (FP + TN)
    TPR = 0.0 if TP + FN == 0.0 else TP / (TP + FN)
    return TPR, FPR

def hacer_pepinillo(modelo, nombre, test=False):
    """
    Descripción: Guarda un modelo entrenado en formato pickle. La ubicación depende del entorno (test o producción), siendo por default la ruta fija de modelos.
    Entrada - modelo (objeto): Modelo entrenado a guardar.
              nombre (str): Nombre del archivo destino.
              test (bool): Indica si se trata de un entorno de prueba (True) o producción (False).
    Salida: Ninguna.
    """
    if test:
        os.makedirs(os.path.dirname(nombre), exist_ok=True)
        with open(nombre, "wb") as f:
            pickle.dump(modelo, f)
    else:
        with open(f"../../models/{nombre}", "wb") as f:
            pickle.dump(modelo, f)
