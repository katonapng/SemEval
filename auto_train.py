# %%
import os
import random
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.learner import DataLoaders
from fastai.metrics import RocAuc
from fastai.optimizer import OptimWrapper
from fastai.vision.all import Learner
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (BertModel, BertTokenizer, DistilBertModel,
                          DistilBertTokenizerFast, RobertaModel,
                          RobertaTokenizer)

# %%
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
# Fix the random state
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# If using GPU
torch.cuda.manual_seed_all(42)

# Optionally, set deterministic behavior in PyTorch to reduce randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
data_translated = pd.read_csv("df_translated.csv")

# %%
TEST_SIZE = 0.85
labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# %%
with_attention = True
# with_attention = False

with_pos_weights = True
# with_pos_weights = False

# pretrained_model = "distilbert"
pretrained_model = "bert"
# pretrained_model = "roberta"

model_config = {
    "distilbert": {
        "tokenizer": DistilBertTokenizerFast,
        "model": DistilBertModel,
        "model_name": "distilbert-base-uncased"
    },
    "bert": {
        "tokenizer": BertTokenizer,
        "model": BertModel,
        "model_name": "bert-base-uncased"
    },
    "roberta": {
        "tokenizer": RobertaTokenizer,
        "model": RobertaModel,
        "model_name": "FacebookAI/roberta-base"
    }
}

config = model_config.get(pretrained_model)

if config:
    tokenizer = config["tokenizer"]
    model_to_use = config["model"]
    model_name = config["model_name"]
else:
    raise ValueError(f"Model {pretrained_model} not supported!")

print(f"Using {model_name} with tokenizer {tokenizer}")

# %%
which_data = "original_eng" # only origina dataset
# which_data = "original_backtranslated_eng" # original plus backtranslated
# which_data = "translated" # original and translated from different languages
# which_data = "translated_backtranslated" # original, backtranslated and translated from different languages

if which_data == "original_eng":
    df = data_translated[data_translated["comment"] == "original_eng"]
if which_data == "original_backtranslated_eng":
    df = data_translated[(data_translated["comment"] == "original_eng") | (data_translated["comment"] == "backtranslate_de")]
if which_data == "translated":
    df = data_translated[(data_translated["comment"] != "backtranslate_de") & data_translated.notna()]
if which_data == "translated_backtranslated":
    df = data_translated[data_translated.notna()]

# %%
df["comment"].unique()

# %%
language_label_distribution = df.groupby(['comment'])[labels].sum()


plt.figure(figsize=(8, 6))
ax = language_label_distribution.plot(kind='bar', figsize=(8, 6), width=0.8, colormap="viridis")

plt.title("Label Frequency Distribution Across Languages")
plt.xlabel("Language")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.legend(title="Labels")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# %%
MAX_LEN = df.text.str.len().max()
print(f"Max length of text: {MAX_LEN}")

# %%
# Dataset definition
class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.texts = data["text"].tolist()
        self.labels = data[["Anger", "Fear", "Joy", "Sadness", "Surprise"]].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        encodings = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        return (input_ids, attention_mask), labels

# %%
class ModelMultiLabel(torch.nn.Module):
    def __init__(self, model, model_name, num_labels):
        super(ModelMultiLabel, self).__init__()
        self.bert = model.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, x):
        input_ids, attention_mask = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        return self.fc(pooled_output)
    

# %%
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)  # Apply softmax across the sequence length
        )

    def forward(self, hidden_states, attention_mask):
        # Apply attention mask to ignore [PAD] tokens
        weights = self.attention(hidden_states)
        weights = weights * attention_mask.unsqueeze(-1)  # Apply mask

        # Normalize attention weights to sum to 1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

        # Weighted sum of hidden states
        pooled_output = torch.sum(weights * hidden_states, dim=1)
        return pooled_output, weights



# model + Multi-Head Attention Pooling for Multi-label Emotion Classification
class ModelAttentionMultiLabel(torch.nn.Module):
    def __init__(self, model, model_name,  num_labels):
        super(ModelAttentionMultiLabel, self).__init__()
        self.bert = model.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        hidden_size = self.bert.config.hidden_size  # Typically 768

        # Attention Pooling Layer
        self.attention_pooling = AttentionPooling(hidden_size)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        input_ids, attention_mask = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Apply Attention Pooling
        pooled_output, attention_weights = self.attention_pooling(hidden_states, attention_mask)

        # Classification
        logits = self.classifier(pooled_output)

        return logits

# %%
train_data, valid_data = train_test_split(df, test_size=TEST_SIZE, random_state=42)
tokenizer = tokenizer.from_pretrained(model_name)

# %%
train_dataset = EmotionDataset(
    train_data,
    tokenizer
)
valid_dataset = EmotionDataset(
    valid_data,
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# %%
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
# If CUDA isn't available, check for MPS (Metal Performance Shaders)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    # Fall back to CPU if neither CUDA nor MPS is available
    device = torch.device("cpu")

print(f"Using device: {device}") 

# Define model and training components
if with_attention:
    model = ModelAttentionMultiLabel(model_to_use, model_name, len(labels)).to(device)
else:
    model = ModelMultiLabel(model_to_use, model_name,len(labels)).to(device)


# Compute pos_weight for BCEWithLogitsLoss 
if with_pos_weights:
    values = train_data[labels].values.tolist()
    labels_tensor = torch.tensor(values, dtype=torch.float)
    num_positives = labels_tensor.sum(dim=0)
    num_negatives = labels_tensor.shape[0] - num_positives
    pos_weight = num_negatives / num_positives
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)

    # Define the loss function
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
else:
    loss_func = torch.nn.BCEWithLogitsLoss()

# %%
# Add timestamp to the directory name
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
name = f"{pretrained_model}_attention_{with_attention}_pos_weights_{with_pos_weights}_{which_data}"
output_path = Path(f'model_{name}_{timestamp}')
output_path.mkdir(exist_ok=True, parents=True)

# FastAI DataLoaders
dls = DataLoaders(train_loader, valid_loader, device=device)

# Define Learner
learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=partial(OptimWrapper, opt=torch.optim.AdamW),
    metrics=[RocAuc()],
    path=output_path
)

# Callbacks
cbs = [
    SaveModelCallback(monitor='valid_loss', fname='best_valid'),
    EarlyStoppingCallback(monitor='valid_loss', patience=9),
    CSVLogger()
]

# %%
learn.fit_one_cycle(n_epoch=1, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)

# %%
learn.recorder.plot_loss()
plt.show()

# %%
# save learn model in pth format
learn.save(f'model_last')


# %%
df_eval = pd.read_csv('public_data_test/track_a/dev/eng.csv')
# dev_df = valid_data
df_eval.head()


# %%
# Function to preprocess and predict
def predict_classes(learner, texts, tokenizer, max_len=128):
    predictions = []

    learner.model.eval()
    with torch.no_grad():
        for text in texts:
            # Tokenize and prepare input
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Make predictions
            logits = learner.model((input_ids, attention_mask))
            probs = torch.sigmoid(logits)  # Multi-label classification
            predicted_labels = (probs > 0.5).int().tolist()[0]  # Binary predictions
            # predicted_labels = (logits > 0).int().tolist()[0]  # Binary predictions

            predictions.append(predicted_labels)
    return predictions


# Load best model
best_model_path = output_path / 'models' / 'best_valid.pth'
learn.load(best_model_path.stem)


# Apply the model to the dataset
# drop dev_df with missing text
df_eval = df_eval.dropna(subset=['text'])
texts = df_eval["text"].tolist()  # Handle missing texts
predictions = predict_classes(learn, texts, tokenizer)

# Add predictions to the DataFrame
emotion_labels = ["pred_anger", "pred_fear", "pred_joy", "pred_sadness", "pred_surprise"]
prediction_df = pd.DataFrame(predictions, columns=emotion_labels)
df_eval[emotion_labels] = prediction_df.values

# Save the true and predicted labels as string lists
# dev_df["True Labels"] = dev_df[["Anger", "Fear", "Joy", "Sadness", "Surprise"]].values.tolist()
df_eval["True Labels"] = df_eval[["anger", "fear", "joy", "sadness", "surprise"]].values.tolist()

df_eval["Predicted Labels"] = df_eval[emotion_labels].values.tolist()

# Convert lists to strings
df_eval["True Labels"] = df_eval["True Labels"].apply(lambda x: str(x))
df_eval["Predicted Labels"] = df_eval["Predicted Labels"].apply(lambda x: str(x))

# Save the updated DataFrame to a new file
results_path = Path(f"results_{name}_{timestamp}")
results_path.mkdir(exist_ok=True, parents=True)
output_csv = f"{results_path}/dev_predictions.csv"

df_eval.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")

# %%



