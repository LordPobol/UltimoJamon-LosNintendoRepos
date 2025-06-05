from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

import numpy as np

import os

MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"
BATCH_SIZE = 16
RANDOM_STATE = 22
NUM_EPOCHS = 50
KFOLDS = 5
PATIENCE = 5
TARGET_AUC = 0.93
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
    

class RobertaClassifier(nn.Module):
    def __init__(self, unfreeze_layers=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            
        if unfreeze_layers:
            for name, param in self.bert.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        self.intermediate = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.mean(outputs.last_hidden_state, dim=1) # CLS token = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(x)
        x = self.relu(self.intermediate(x))
        x = self.classifier(x)
        return x.squeeze(-1)
    
def generar_loader(X, y, shuffle=None):
    ds = TweetDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return loader

def obtener_probs(X, y, modelo, y_verdaderas=[], y_probs=[]):
    loader = generar_loader(X, y)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            outputs = modelo(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()

            y_probs.extend(probs)
            y_verdaderas.extend(labels)

    y_true = np.array(y_verdaderas)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    return y_true, y_probs, y_pred

def cargar_estado_modelo():
    modelo_RoBERTa = RobertaClassifier()
    modelo_RoBERTa.load_state_dict(
        torch.load(
            os.path.abspath("../../models/final_best_model.pt"),
            map_location=torch.device('cpu')
        )
    )
    return modelo_RoBERTa