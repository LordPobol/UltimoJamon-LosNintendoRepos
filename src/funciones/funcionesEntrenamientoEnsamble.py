# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de una parte de la fase de entrenamiento de ensamble.
# 
# --------------------------------------------------

# Librerías de cálculo y manejo de datos
import numpy as np
import joblib

# Librería para realizar un stratified k-fold
from sklearn.model_selection import StratifiedKFold

# Importación de los modelos que componen el ensamblado
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


#######################################################################
# Variables de entorno #
########################

# Diccionario con los modelos base
model_classes = {
    'MLP': MLPClassifier,
    'PAC': PassiveAggressiveClassifier,
    'RF': RandomForestClassifier,
    'SVM': SVC,
    'XGB': XGBClassifier,
}

# Diccionario con los mejores parámetros de los modelos base
best_params_dict = {
    'MLP': {
        'activation': 'logistic',
        'hidden_layer_sizes': (400, 300, 200, 100),
        'learning_rate': 'adaptive',
        'max_iter': 50,
        'solver': 'adam',
        'random_state': 22
    },
    'PAC': {
        'C': 0.001,
        'loss': 'squared_hinge',
        'max_iter': 200,
        'random_state': 22
    },
    'RF': {
        'criterion': 'entropy',
        'max_depth': None,
        'max_features': 'log2',
        'min_samples_leaf': 1,
        'min_samples_split': 7,
        'n_estimators': 400,
        'random_state': 22
    },
    'SVM': {
        'C': 350,
        'gamma': 0.001,
        'kernel': 'rbf',
        'probability': True,
        'random_state': 22
    },
    'XGB': {
        'eval_metric': 'logloss',
        'learning_rate': 0.25,
        'max_depth': 1,
        'n_estimators': 600,
        'subsample': 1,
        'random_state': 22
    },
}

# Cantidad de modelos utilizados para ensamblar
n_models = len(model_classes)
#######################################################################



#######################################################################
# Funciones para el entrenamiento del ensamblado #
##################################################

def parametros_4_oof(X_train, n_splits=5):
    """
    Descripción: Prepara los objetos necesarios para la validación cruzada estratificada y la generación de predicciones OOF (Out-Of-Fold).
    Entrada: - X_train (DataFrame): Conjunto de entrenamiento (atributos).
             - n_splits (int): Número de particiones para la validación cruzada (por defecto 5).
    Salida:
        - skf (StratifiedKFold): Objeto de validación cruzada estratificada.
        - oof_preds (ndarray): Matriz inicializada con ceros para almacenar las predicciones OOF de cada modelo.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Encontramos que un random state de 42 muestra un mayor desempeño
    oof_preds = np.zeros((X_train.shape[0], n_models))

    return skf, oof_preds

def predicciones_de_modelos(X_train, y_train, skf, oof_preds):
    """
    Descripción: Entrena múltiples modelos usando validación cruzada y genera predicciones OOF (Out-Of-Fold) para cada uno.
    Entrada: - X_train (DataFrame o Series): Atributos del conjunto de entrenamiento.
             - y_train (Series): Etiquetas del conjunto de entrenamiento.
             - skf (StratifiedKFold): Objeto de validación cruzada estratificada.
             - oof_preds (ndarray): Matriz para almacenar las predicciones OOF de cada modelo.
    Salida: - oof_preds (ndarray): Matriz con las predicciones OOF generadas por cada modelo entrenado.
    """
    for idx, (name, model_class) in enumerate(model_classes.items()):
        print(f"Generando predicciones OOF para el modelo de {name}...")
        temp_oof = np.zeros(X_train.shape[0])
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]

            model = model_class(**best_params_dict[name])
            model.fit(X_tr, y_tr)
            if name != 'PAC':
                temp_oof[val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                temp_oof[val_idx] = model.decision_function(X_val)

        oof_preds[:, idx] = temp_oof
    
    return oof_preds

def probar_ensamble(X_test):
    """
    Descripción: Genera las predicciones de los modelos previamente entrenados sobre el conjunto de prueba, 
                 creando una matriz de características meta para el modelo de ensamblado.
    Entrada: - X_test (DataFrame o Series): Conjunto de prueba (atributos).
    Salida: - X_meta_test (ndarray): Matriz con las predicciones de todos los modelos sobre el conjunto de prueba,
                                     que será utilizada como entrada para el modelo ensamblador (meta-modelo).
    """
    X_meta_test = np.zeros((X_test.shape[0], n_models))

    for idx, name in enumerate(model_classes.keys()):
        model = joblib.load(f"../../models/model{name}.pkl")
        if name != 'PAC':
            X_meta_test[:, idx] = model.predict_proba(X_test)[:, 1]
        else:
            X_meta_test[:, idx] = model.decision_function(X_test)

    return X_meta_test