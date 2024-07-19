import librosa
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torchmetrics
import os
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 20
    LR = 3e-4
    # Others
    SEED = 42

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED) # Seed 고정
df = pd.read_csv('train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

# def get_mfcc_feature(df, train_mode=True):
#     features = []
#     labels = []
#     for _, row in tqdm(df.iterrows()):
#         # librosa패키지를 사용하여 wav 파일 load
#         y, sr = librosa.load(row['path'], sr=CONFIG.SR)
#
#         # librosa패키지를 사용하여 mfcc 추출
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
#         mfcc = np.mean(mfcc.T, axis=0)
#         features.append(mfcc)
#
#         if train_mode:
#             label = row['label']
#             label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
#             label_vector[0 if label == 'fake' else 1] = 1
#             labels.append(label_vector)
#
#     if train_mode:
#         return features, labels
#     return features
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)

        if train_mode:
            # Data augmentation
            if random.random() < 0.5:
                y = add_noise(y)
            if random.random() < 0.5:
                y = pitch_shift(y, sr, random.uniform(-2, 2))
            if random.random() < 0.5:
                y = time_stretch(y, random.uniform(0.8, 1.2))

        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        mfcc = np.mean(mfcc.T, axis=0)
        chroma = np.mean(chroma.T, axis=0)
        spec_contrast = np.mean(spec_contrast.T, axis=0)
        tonnetz = np.mean(tonnetz.T, axis=0)
        rms = np.mean(rms.T, axis=0)

        feature_vector = np.concatenate((mfcc, chroma, spec_contrast, tonnetz, rms))
        features.append(feature_vector)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features
train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, output_dim=Config.N_CLASSES):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        output = self.fc(h_lstm[:, -1, :])
        return output


class EnsembleModel(nn.Module):
    def __init__(self, cnn_model, lstm_model):
        super(EnsembleModel, self).__init__()
        self.cnn = cnn_model
        self.lstm = lstm_model

    def forward(self, x):
        cnn_output = self.cnn(x)
        lstm_output = self.lstm(x)
        combined = torch.cat((cnn_output, lstm_output), dim=1)
        return combined
from sklearn.metrics import roc_auc_score

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)

    best_val_score = 0
    best_model = None

    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            output = model(features)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)

            probs = model(features)

            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)

    return _val_loss, auc_score
input_dim = 39 # mfcc + chroma + spectral contrast + tonnetz

# Training the ensemble model
cnn_model = MLP(input_dim=input_dim)
lstm_model = BiLSTM()
ensemble_model = EnsembleModel(cnn_model, lstm_model)
optimizer = torch.optim.Adam(params=ensemble_model.parameters(), lr=CONFIG.LR)
infer_model = train(ensemble_model, optimizer, train_loader, val_loader, device)

test = pd.read_csv('./test.csv')
test_mfcc = get_mfcc_feature(test, False)
test_dataset = CustomDataset(test_mfcc, None)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)

            probs = model(features)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions


preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.head()

submit.to_csv('./baseline_submit.csv', index=False)