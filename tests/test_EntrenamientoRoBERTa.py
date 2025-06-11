import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from funciones.funcionesEntrenamientoRoBERTa import *

####################################################################
################### Datos simulados para pruebas ###################
tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
texts = ["Esto es un tweet positivo.", "Este tweet es negativo."]
labels = [1, 0]

# Tests para dataset dummy
@pytest.fixture
def dummy_dataset():
    return TweetDataset(texts, labels)

def test_tweet_dataset_len(dummy_dataset):
    assert len(dummy_dataset) == 2

def test_tweet_dataset_getitem(dummy_dataset):
    sample = dummy_dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample
    assert isinstance(sample['input_ids'], torch.Tensor)
    assert sample['labels'].item() == 1
####################################################################
####################################################################


""" Pruebas unitarias para las funciones de entrenamiento del modelo RoBERTa """

#################################################
# Pruebas unitarias para la clase: TweetDataset #
#################################################

def test_tweet_dataset_len():
    dataset = TweetDataset(texts, labels)
    assert len(dataset) == 2

def test_tweet_dataset_item_format():
    dataset = TweetDataset(texts, labels)
    sample = dataset[0]
    assert "input_ids" in sample and "attention_mask" in sample and "labels" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)

######################################################
# Pruebas unitarias para la clase: RobertaClassifier #
######################################################

def test_roberta_classifier_forward():
    model = RobertaClassifier(unfreeze_layers=['encoder.layer.11'])
    model.eval()
    dummy_input_ids = torch.randint(0, 100, (2, 128))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    with torch.no_grad():
        output = model(dummy_input_ids, dummy_attention_mask)
    assert output.shape == (2,)
    assert isinstance(output, torch.Tensor)


def test_roberta_classifier_forward_shape():
    model = RobertaClassifier()
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == len(texts)

######################################################################################
# Pruebas unitarias para la primera función: generar_loader(Dataset, Dataset, model) #
######################################################################################

def test_generar_loader():
    loader = generar_loader(texts, labels, shuffle=False)
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert 'input_ids' in batch and 'attention_mask' in batch and 'labels' in batch

#####################################################################################
# Pruebas unitarias para la segunda función: obtener_probs(Dataset, Dataset, model) #
#####################################################################################

@patch("funciones.funcionesEntrenamientoRoBERTa.generar_loader")
def test_obtener_probs(mock_loader):
    # Creamos un batch simulado
    dummy_input = torch.randint(0, 100, (2, 10))
    dummy_labels = torch.tensor([1.0, 0.0])
    mock_batch = {
        "input_ids": dummy_input,
        "attention_mask": dummy_input,
        "labels": dummy_labels
    }
    mock_loader.return_value = [mock_batch]

    # Creamos un modelo simulado
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return torch.tensor([0.8, 0.2])

    modelo = DummyModel()

    y_true, y_probs, y_pred = obtener_probs(texts, labels, modelo)

    np.testing.assert_array_equal(y_true, np.array([1.0, 0.0]))
    np.testing.assert_allclose(y_probs, np.array([0.689974, 0.549834]), atol=1e-2)
    np.testing.assert_array_equal(y_pred, np.array([1, 1]))

def test_obtener_probs_dimensiones():
    # Creamos un modelo de prueba
    model = RobertaClassifier()
    model.eval()

    # Usamos los mismos datos dummy
    y_true, y_probs, y_pred = obtener_probs(texts, labels, model)

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_probs, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

    assert y_true.shape == y_probs.shape == y_pred.shape
    assert set(np.unique(y_pred)).issubset({0, 1})

#####################################################################
# Pruebas unitarias para la tercera función: cargar_estado_modelo() #
#####################################################################

@patch("torch.load")
@patch("funciones.funcionesEntrenamientoRoBERTa.RobertaClassifier")
def test_cargar_estado_modelo(mock_model_class, mock_torch_load):
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model

    mock_state = {"layer": torch.tensor([1, 2, 3])}
    mock_torch_load.return_value = mock_state

    model = cargar_estado_modelo()
    mock_model.load_state_dict.assert_called_once_with(mock_state)
    assert model == mock_model

#################################################################################
# Pruebas unitarias para la cuarta función, primera de embeddings: data4embed() #
#################################################################################

@patch("funciones.funcionesEntrenamientoRoBERTa.pd.read_csv")
def test_data4embed_train(mock_read_csv):
    mock_df = pd.DataFrame({
        "texto_bert": ["texto1", "texto2"],
        "class": [0, 1]
    })
    mock_read_csv.return_value = mock_df

    texts, labels = data4embed(test=False)

    assert texts == ["texto1", "texto2"]
    assert labels == [0, 1]
    mock_read_csv.assert_called_once_with("../../data/ds_BETO.csv")

@patch("funciones.funcionesEntrenamientoRoBERTa.pd.read_csv")
def test_data4embed_test(mock_read_csv):
    mock_df = pd.DataFrame({
        "texto_bert": ["test1", "test2", "test3"],
        "class": [1, 0, 1]
    })
    mock_read_csv.return_value = mock_df

    texts, labels = data4embed(test=True)

    assert texts == ["test1", "test2", "test3"]
    assert labels == [1, 0, 1]
    mock_read_csv.assert_called_once_with("../../data/ds_BETO_TEST_FINAL.csv")

@patch("funciones.funcionesEntrenamientoRoBERTa.pd.read_csv")
def test_data4embed_empty(mock_read_csv):
    mock_df = pd.DataFrame(columns=["texto_bert", "class"])
    mock_read_csv.return_value = mock_df

    texts, labels = data4embed(test=False)

    assert texts == []
    assert labels == []

#####################################################################################
# Pruebas unitarias para la quinta función, segunda de embeddings: get_embeddings() #
#####################################################################################

@patch("funciones.funcionesEntrenamientoRoBERTa.model2embed")
@patch("funciones.funcionesEntrenamientoRoBERTa.tokenizer")
def test_get_embeddings_basic(mock_tokenizer, mock_model):
    # Simula tokenizer
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    
    # Simula salida del modelo
    dummy_hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape: (1, 2, 2)
    mock_output = MagicMock()
    mock_output.last_hidden_state = dummy_hidden
    mock_model.return_value = mock_output

    text_list = ["Hola mundo"]
    embeddings = get_embeddings(text_list)

    expected_mean = dummy_hidden.mean(dim=1).squeeze().numpy()
    np.testing.assert_array_almost_equal(embeddings[0], expected_mean)

@patch("funciones.funcionesEntrenamientoRoBERTa.model2embed")
@patch("funciones.funcionesEntrenamientoRoBERTa.tokenizer")
def test_get_embeddings_multiple_inputs(mock_tokenizer, mock_model):
    mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    
    dummy_output = MagicMock()
    dummy_output.last_hidden_state = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])  # mean = [2.0, 2.0]
    mock_model.return_value = dummy_output

    text_list = ["t1", "t2"]
    embeddings = get_embeddings(text_list)

    assert embeddings.shape == (2, 2)
    np.testing.assert_array_equal(embeddings, np.array([[2.0, 2.0], [2.0, 2.0]]))

@patch("funciones.funcionesEntrenamientoRoBERTa.model2embed")
@patch("funciones.funcionesEntrenamientoRoBERTa.tokenizer")
def test_get_embeddings_empty_input(mock_tokenizer, mock_model):
    text_list = []
    embeddings = get_embeddings(text_list)
    assert embeddings.shape == (0,)
