# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de la fase de entrenamiento de LLM.
# 
# --------------------------------------------------

import pandas as pd
from tqdm import tqdm
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
    df = pd.read_csv("../../data/data_train.csv")
    # Separar características y etiquetas
    X = df["tweet_text"]
    y = df["class"].map({'control': 0, 'anorexia': 1})
    return X, y

def imprimir_forma(df):
    """
    Descripción: Retorna las dimensiones del DataFrame y las primeras 5 filas para una inspección rápida.
    Entrada - df (DataFrame): Conjunto de datos a inspeccionar.
    Salida  - shape (tuple): Dimensiones del DataFrame.
            - head (DataFrame): Primeras 5 filas del DataFrame.
    """
    return df.shape, df.head(5)

def obtener_predicciones(chain, X):
    """
    Descripción: Obtiene las predicciones de nuestro LLM usado para clasificación, recibiendo el pipeline.
    Entrada - chain (Chain): Pipeline del modelo recibido.
            - X (DataFrame): Conjunto de datos a clasificar.
    Salida  - predicciones (Series): Lista de predicciones.
    """
    predicciones = []
    for tweet in tqdm(X, desc="Clasificando tweets con LLaMA3"):
        result = chain.invoke({"tweet": tweet})
        answer = result['text'].strip().lower()
        if "Sí" in answer or "sí" in answer:
            predicciones.append(1)
        elif "No" in answer or "no" in answer:
            predicciones.append(0)
        else:
            print("Error en tweet:", result['text'])
            break  # debug en caso de error
    return predicciones

def reporte_clasificacion(predicts, y_real):
    """
    Descripción: Generar predicciones, calcular probabilidades de predicción o funciones de decisión, y crea un reporte de clasificación.
    Entrada - predicts (array): Lista de predicciones.
            - y_real (array): Conjunto de las etiquetas verdaderas.
    Salida - y_pred (array): Etiquetas predichas.
           - reporte (str): Métricas de desempeño del modelo.
    """
    y_preds = pd.DataFrame(predicts, columns=['predicciones'])
    reporte = classification_report(y_real, y_preds)
    return y_preds, reporte

def crear_matriz_confusion(y, y_pred):
    """
    Descripción: Calcula y prepara los valores  de la matriz de confusión para su visualización.
    Entrada - y (array): Etiquetas verdaderas.
            - y_pred (array): Etiquetas predichas.
    Salida - cm (array): Matriz de confusión en forma de matriz.
           - disp (objeto): Objeto para mostrar la matriz de confusión.
    """
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return cm, disp

def calcular_roc_auc(y, y_pred):
    """
    Descripción: Calcula la curva ROC y el valor AUC del modelo.
    Entrada - y (array): Etiquetas verdaderas.
            - y_pred (array): Predicciones del modelo.
    Salida - fpr (array): Tasas de falsos positivos.
           - tpr (array): Tasas de verdaderos positivos.
           - thresholds (array): Umbrales de decisión.
           - auc_score (float): Área bajo la curva ROC.
    """
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    auc_score = roc_auc_score(y, y_pred)
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