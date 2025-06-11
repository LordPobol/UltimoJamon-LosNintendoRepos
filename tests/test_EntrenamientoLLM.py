# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Archivo de pruebas de la fase de uso del LLM con cobertura del 95%.
# 
# --------------------------------------------------

import pandas as pd
import sys
import os
from unittest.mock import MagicMock
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from funciones.funcionesEntrenamientoLLM import *


""" Pruebas unitarias para las funciones de entrenamiento del comportamiento del LLM """

##########################################################################
# Pruebas unitarias para la primera función: cargar_datos_entrenamiento()#
##########################################################################

def test_cargar_datos_entrenamiento(monkeypatch):
    data = pd.DataFrame({
        "tweet_text": ["me encanta comer", "no comí nada"],
        "class": ["control", "anorexia"]
    })

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: data)
    X, y = cargar_datos_entrenamiento()

    assert isinstance(X, pd.Series)
    assert isinstance(y, pd.Series)
    assert list(X) == ["me encanta comer", "no comí nada"]
    assert list(y) == [0, 1]

def test_cargar_datos_entrenamiento_monkeypatch(monkeypatch):
    # Simular archivo CSV con datos controlados
    dummy_data = pd.DataFrame({
        "tweet_text": ["comí poco", "me encanta comer con mi familia"],
        "class": ["anorexia", "control"]
    })

    # Sustituir pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: dummy_data)

    X, y = cargar_datos_entrenamiento()

    assert list(X) == ["comí poco", "me encanta comer con mi familia"]
    assert list(y) == [1, 0]
    assert isinstance(X, pd.Series)
    assert isinstance(y, pd.Series)

########################################################################
# Pruebas unitarias para la segunda función: imprimir_forma(Dataframe) #
########################################################################

def test_imprimir_forma():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12]
    })
    shape, head = imprimir_forma(df)

    assert shape == (6, 2)
    assert head.equals(df.head(5))

def test_imprimir_forma_vacía():
    df = pd.DataFrame({
        "a": [],
        "b": []
    })
    shape, head = imprimir_forma(df)

    assert shape == (0, 2)
    assert head.equals(df.head(5))

########################################################################################
# Pruebas unitarias para la tercera función: obtener_predicciones(pipeline, Dataframe) #
########################################################################################

def test_obtener_predicciones_si_y_no():
    class FakeChain:
        def invoke(self, x):
            tweet = x["tweet"]
            if "comí en todo el día" in tweet:
                return {"text": "Sí"}
            else:
                return {"text": "No"}

    X = pd.Series(["no comí en todo el día", "hoy comí con amigos"])
    result = obtener_predicciones(FakeChain(), X)
    assert result == [1, 0]

def test_obtener_predicciones_con_error(capfd):
    class FakeChain:
        def invoke(self, x):
            return {"text": "Tal vez"}

    X = pd.Series(["tweet que da error"])
    preds = obtener_predicciones(FakeChain(), X)
    
    out, err = capfd.readouterr()
    assert "Error en tweet" in out

def test_obtener_predicciones_con_fake_chain():
    # FakeChain que responde "Sí" si el tweet menciona "sin comer"
    class FakeChain:
        def invoke(self, x):
            return {"text": "Sí"} if "sin comer" in x["tweet"] else {"text": "No"}

    X = pd.Series(["sin comer desde ayer", "hoy desayuné bien"])
    predicciones = obtener_predicciones(FakeChain(), X)

    assert predicciones == [1, 0]

def test_obtener_predicciones_con_texto_invalido_y_capfd(capfd):
    # FakeChain que da un texto inesperado
    class FakeChain:
        def invoke(self, x):
            return {"text": "Respuesta inválida"}

    X = pd.Series(["mensaje extraño sin contexto"])
    predicciones = obtener_predicciones(FakeChain(), X)

    out, err = capfd.readouterr()

    assert "Error en tweet:" in out
    assert isinstance(predicciones, list)
    assert len(predicciones) == 0  # El bucle se rompe en el primer error

def test_obtener_predicciones_magicmock():
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [
        {"text": "Sí"},
        {"text": "No"},
        {"text": "Sí"}
    ]
    X = pd.Series(["tweet 1", "tweet 2", "tweet 3"])
    preds = obtener_predicciones(mock_chain, X)

    assert preds == [1, 0, 1]
    assert mock_chain.invoke.call_count == 3

####################################################################################
# Pruebas unitarias para la cuarta función: reporte_clasificacion_llm(list, list) #
####################################################################################

def test_reporte_clasificacion_llm():
    y_real = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]

    df_preds, reporte = reporte_clasificacion_llm(y_pred, y_real)

    assert isinstance(df_preds, pd.DataFrame)
    assert list(df_preds['predicciones']) == y_pred
    assert "precision" in reporte
    assert "recall" in reporte

from sklearn.metrics import classification_report

def test_reporte_clasificacion_llm_con_precision():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]

    df_pred, reporte = reporte_clasificacion_llm(y_pred, y_true)

    assert isinstance(df_pred, pd.DataFrame)
    assert "precision" in reporte
    assert "recall" in reporte
    assert df_pred.shape == (4, 1)
