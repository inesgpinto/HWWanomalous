import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import math



train_features = TRAIN_FEATURES



print("Reading data")
data = pd.read_hdf("data/data_ml_2class.h5", key='table',mode='r')

print("Importing model")
nn = load_model(f"models/nn_2class_cos", compile=False)
nn_scaler = pickle.load(open(f"models/nn_2class_cosscaler.pck", 'rb'))

nn.summary()

data_train   = data.query(" gen_split == 'train'  ")
data_val = data.query("gen_split == 'val'")

X = nn_scaler.transform(data[train_features].values)
print("Predictions")
score = nn.predict(X)
data["y_pred"] = score





print("Plotting scores")
bins = np.linspace(0.35,1,30)
plt.figure(figsize=(8, 6))
bins = plt.hist(
    data.query(f'gen_split == "train" & category=="bkg" ')["y_pred"],
    weights = data.query(f'gen_split == "train" & category=="bkg" ')['train_weight'],
    bins = bins,
    density=True, 
    histtype='stepfilled', 
    color = 'blue',
    alpha = 0.5, 
    label = "bkg train"
)
plt.hist(
    data.query(f'gen_split == "val" & category=="bkg" ')["y_pred"],
    weights = data.query(f'gen_split == "val" & category=="bkg" ')['train_weight'],
    bins = bins[1],
    density=True, 
    histtype='step', 
    color = 'blue',
    label = "bkg val"
)
bins = plt.hist(
    data.query(f'gen_split == "train" & category=="signal" ')["y_pred"],
    weights = data.query(f'gen_split == "train" & category=="signal" ')['train_weight'],
    bins = bins[1],
    density=True, 
    histtype='stepfilled', 
    color = 'red',
    alpha = 0.5, 
    label = "signal train"
)
plt.hist(
    data.query(f'gen_split == "val" & category=="signal" ')["y_pred"],
    weights = data.query(f'gen_split == "val" & category=="signal" ')['train_weight'],
    bins = bins[1],
    density=True, 
    histtype='step', 
    color = 'red',
    label = "signal val"
)
plt.xlabel("DNN output")
plt.ylabel("Density (A.U.)")
plt.legend()
plt.savefig(f'2output/NN_OUTPUT.png', transparent=False)
plt.yscale('log')
plt.savefig(f'2output/NN_OUTPUT_log.png', transparent=False)
plt.close()





split = 'val'
fpr, tpr, _ = roc_curve(data.query(f'gen_split == "{split}"')['label'], data.query(f'gen_split == "{split}"')['y_pred'],
                        sample_weight=data.query(f'gen_split == "{split}"')['train_weight'])
auc = roc_auc_score(data.query(f'gen_split == "{split}"')['label'], data.query(f'gen_split == "{split}"')['y_pred'],
                    sample_weight=data.query(f'gen_split == "{split}"')['train_weight'])
plt.plot(fpr,tpr,label=f'AUC {auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC curve')
plt.savefig('2output/ROC.png')
plt.show()

def significance (s,b):

  sig = math.sqrt(2*(s+b)*math.log(1+s/b)-2*s)
  return sig if not (np.isnan(sig) or np.isinf(sig)) else 0
plt.clf()
Ncuts = 40
cuts = np.linspace(0,1,Ncuts) 
sig  = np.zeros_like(cuts)

for i,cut in enumerate(cuts):

  s = data.query(f'(gen_split == "{split}") & (label==1) & (y_pred>{cut})')['train_weight'].sum()
  b = data.query(f'(gen_split == "{split}") & (label==0) & (y_pred>{cut})') ['train_weight'].sum() 
  sig[i] = significance(s,b)
  #print(f'cut {cut} index {i} signal {s} background {b} significance {sig[i]}')

plt.plot(cuts,sig,label=f'max {max(sig)}')
plt.xlabel('DNN output')
plt.ylabel('Significance')
plt.legend()
plt.savefig('2output/significance.png')
plt.show()

corr_matrix = False

if corr_matrix:

    for split in data["gen_split"].unique():
        y_true = data.query(f'gen_split == "{split}"')['label']
        y_pred = data.query(f'gen_split == "{split}"')['y_pred']
        w = data.query(f'gen_split == "{split}"')['train_weight']
        cm = confusion_matrix(y_true, y_pred, 
                            normalize = 'all')

        labels = ['SM', 'BSM']

        cm_df = pd.DataFrame(cm,index=labels, columns = labels,
                            sample_weight = w)

        

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"2output/corr_matrix_{split}.png")
        plt.show()
