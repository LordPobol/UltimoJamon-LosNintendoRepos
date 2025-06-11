# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Archivo de pruebas de la fase de entrenamiento del ensamble con cobertura del 90%.
# 
# --------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pytest
import joblib
import sys
import os

# Importación de los modelos que componen el ensamblado
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from funciones.funcionesEntrenamientoEnsamble import *

######################################################################
## Cargamos la misma variable que contiene los modelos del ensamble ##
model_classes = {
    'MLP': MLPClassifier,
    'PAC': PassiveAggressiveClassifier,
    'RF': RandomForestClassifier,
    'SVM': SVC,
    'XGB': XGBClassifier,
}
n_models = 5
######################################################################
######################################################################

##################################################################################
## Modelos Dummy para testear funcionalidad de las funciones en casos de prueba ##
class ModeloDummy:
    def __init__(self, tipo='proba'):
        self.tipo = tipo
        self.classes_ = np.array([0, 1])  # necesario para predict_proba

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))

    def decision_function(self, X):
        return np.ones(len(X)) * 0.5

class ModeloDummy2:
    def __init__(self, name):
        self.name = name
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))

    def decision_function(self, X):
        return np.ones(len(X)) * 0.42
##################################################################################
##################################################################################


""" Pruebas unitarias para las funciones de entrenamiento del ensamble """

#########################################################################
# Pruebas unitarias para la primera función: parametros_4_oof(DataFrame)#
#########################################################################

def test_parametros_4_oof_retorna_objetos_correctos():
    X_train = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100)
    })
    skf, oof_preds = parametros_4_oof(X_train, n_splits=5)

    assert isinstance(skf, StratifiedKFold)
    assert isinstance(oof_preds, np.ndarray)
    assert oof_preds.shape == (100, n_models)

def test_parametros_4_oof_con_n_splits_diferente():
    X_train = pd.DataFrame({
        'f1': np.random.rand(50),
        'f2': np.random.rand(50)
    })
    _, oof_preds = parametros_4_oof(X_train, n_splits=3)

    assert oof_preds.shape == (50, n_models)

def test_parametros_4_oof_stratificacion_preservada():
    X_train = pd.DataFrame({
        'feature1': np.random.rand(40),
        'feature2': np.random.rand(40)
    })
    y_train = pd.Series([0]*20 + [1]*20)  # 50% de cada clase

    skf, _ = parametros_4_oof(X_train)

    # Comprobamos que la proporción de clases se mantiene en los splits
    for _, val_idx in skf.split(X_train, y_train):
        y_val = y_train.iloc[val_idx]
        proportion = y_val.value_counts(normalize=True)
        assert all(abs(proportion[cls] - 0.5) < 0.2 for cls in [0, 1])  # tolerancia

####################################################################################################################
# Pruebas unitarias para la segunda función: predicciones_de_modelos(DataFrame, DataFrame, StratifiedKFold, array) #
####################################################################################################################

def test_predicciones_de_modelos_formato_correcto():
    np.random.seed(42)
    X_train = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))

    skf, oof_preds = parametros_4_oof(X_train, n_splits=3)
    oof_result = predicciones_de_modelos(X_train, y_train, skf, oof_preds.copy())

    assert isinstance(oof_result, np.ndarray)
    assert oof_result.shape == (100, n_models)
    # Verificamos que se hayan generado valores diferentes de cero
    assert np.any(oof_result != 0.0)

def test_predicciones_de_modelos_modelos_entrenados_correctamente():
    np.random.seed(10)
    X_train = pd.DataFrame({
        'f1': np.random.rand(60),
        'f2': np.random.rand(60)
    })
    y_train = pd.Series(np.random.randint(0, 2, 60))

    skf, oof_preds = parametros_4_oof(X_train, n_splits=3)
    oof_result = predicciones_de_modelos(X_train, y_train, skf, oof_preds.copy())

    # Revisión básica: todos los modelos deberían haber rellenado algo
    for i in range(n_models):
        assert not np.all(oof_result[:, i] == 0), f"Modelo {i} no generó predicciones"

def test_predicciones_de_modelos_valores_probabilidad_validos():
    X_train = pd.DataFrame({
        'f1': np.random.rand(80),
        'f2': np.random.rand(80)
    })
    y_train = pd.Series(np.random.randint(0, 2, 80))

    skf, oof_preds = parametros_4_oof(X_train)
    oof_result = predicciones_de_modelos(X_train, y_train, skf, oof_preds.copy())

    # Las predicciones deben estar entre 0 y 1 para todos menos PAC (que pueden estar fuera)
    for idx, name in enumerate(model_classes.keys()):
        if name != 'PAC':
            assert np.all((oof_result[:, idx] >= 0) & (oof_result[:, idx] <= 1)), \
                f"Modelo {name} tiene predicciones fuera del rango [0,1]"

#########################################################################
# Pruebas unitarias para la tercera función: probar_ensamble(DataFrame) #
#########################################################################

def test_probar_ensamble_shape(tmp_path, monkeypatch):
    import pandas as pd
    import numpy as np
    import joblib
    from pathlib import Path

    X_test = pd.DataFrame(np.random.rand(10, 5))

    # Guardamos modelos dummy
    for name in model_classes.keys():
        modelo = ModeloDummy(tipo='decision' if name == 'PAC' else 'proba')
        joblib.dump(modelo, tmp_path / f"model{name}.pkl")

    # Guarda la referencia original de joblib.load
    original_load = joblib.load

    # Define la función fake_load, que usa el joblib.load original
    def fake_load(path):
        filename = Path(path).name
        return original_load(tmp_path / filename)

    # Parcha joblib.load con fake_load
    monkeypatch.setattr("joblib.load", fake_load)

    resultado = probar_ensamble(X_test)
    assert resultado.shape == (10, n_models)

def test_probar_ensamble_constantes(tmp_path, monkeypatch):
    X_test = pd.DataFrame(np.random.rand(5, 3))

    # Guardamos los modelos
    for name in model_classes.keys():
        model = ModeloDummy2(name)
        joblib.dump(model, tmp_path / f"model{name}.pkl")

    # Guardar referencia al original antes de hacer monkeypatch
    real_joblib_load = joblib.load
    monkeypatch.setattr("joblib.load", lambda path: real_joblib_load(tmp_path / Path(path).name))

    X_meta = probar_ensamble(X_test)

    for idx, name in enumerate(model_classes.keys()):
        expected = 0.8 if name != 'PAC' else 0.42
        assert np.allclose(X_meta[:, idx], expected), f"Fallo en modelo {name}"

def test_probar_ensamble_modelo_no_encontrado(monkeypatch):
    X_test = pd.DataFrame(np.random.rand(5, 2))

    # Simula que joblib.load lanza FileNotFoundError
    monkeypatch.setattr("joblib.load", lambda path: (_ for _ in ()).throw(FileNotFoundError("Modelo faltante")))

    with pytest.raises(FileNotFoundError, match="Modelo faltante"):
        probar_ensamble(X_test)
