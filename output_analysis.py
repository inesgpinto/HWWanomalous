import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



train_features = TRAIN_FEATURES

print("Reading data")
data = pd.read_hdf("data/data_ml.h5", key='table',mode='r')

print("Importing model")
nn = load_model(f"models/nn_cos", compile=False)
nn_scaler = pickle.load(open(f"models/nn_cosscaler.pck", 'rb'))

nn.summary()

data_train   = data.query(" gen_split == 'train'  ")
data_val = data.query("gen_split == 'val'")

X = nn_scaler.transform(data[train_features].values)
score = nn.predict(X)
score = np.argmax(score,axis=1)
data["y_pred"] = score


for split in data["gen_split"].unique():
    y_true = data.query(f'gen_split == "{split}"')['label']
    y_pred = data.query(f'gen_split == "{split}"')['y_pred']
    cm = confusion_matrix(y_true, y_pred, normalize ='all')

    labels = ['SM', 'CP PAR','CP IMPAR']

    cm_df = pd.DataFrame(cm,index=labels, columns = labels)

    

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt=".2f")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"output/corr_matrix_{split}.png")
    plt.show()
