import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from config import *
import os
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if GPU: 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("Reading file")
data = pd.read_hdf("data/data_ml_2class.h5", key='table',mode='r')

train_features = TRAIN_FEATURES

if FEATURE_COS:
    name = '_cos'
    train_features = []
    for feature in TRAIN_FEATURES:
        if 'phi' in feature:
            data [f'cos_{feature}'] = np.cos( data[feature] )
            train_features.append(f'cos_{feature}')
            
            plt.figure(figsize=(8, 6))
            bins = 50

            bins = plt.hist(data.query(f'sample == "SM"')[f'cos_{feature}'], bins=bins, histtype='step', density=True, color='steelblue', lw=1.5, label='SM')
            plt.hist(data.query(f'sample == "CP IMPAR"')[f'cos_{feature}'], bins=bins[1], histtype='step', density=True, color='purple', lw=1.5, label='CP_odd')
            plt.hist(data.query(f'sample == "CP PAR"')[f'cos_{feature}'], bins=bins[1], histtype='step', density=True, color='green', lw=1.5, label='CP_even')


            plt.xlabel(f'cos_{feature}')
            plt.legend()
            plt.savefig(f'plots/cos_{feature}_histogram.png', transparent=False)
            plt.close()
        else: train_features.append(feature)
else: 
    name = ''


print(train_features)

data_train = data.query(" gen_split == 'train' ")
data_val   = data.query(" gen_split == 'val'   ")

y_train = data_train['label'].values
y_val = data_val['label'].values


X_train = data_train[train_features].values

print("Standard Scaler")
scaler = StandardScaler().fit(X_train)
with open(f'models/nn_2class{name}scaler.pck','wb') as f:
    pickle.dump(scaler,f)

X_train = scaler.transform(data_train[train_features].values)
w_train = data_train['train_weight']

X_val = scaler.transform(data_val[train_features].values)
w_val = data_val['train_weight']

print(X_train.shape, X_val.shape)

print("Define NN_2class and train")



def objective(trial):

    global BEST_VALUE
    global BEST_HISTORY

    n_units = trial.suggest_int("n_units", 15, 256)
    n_layers = trial.suggest_int("n_layers", 5, 20)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1,0.15,0.20,0.25,0.3,0.35,0.4])
    lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)

    model = Sequential()

    for i in range(n_layers):
        model.add(Dense(n_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy', 
              optimizer=optimizer,
              weighted_metrics=['accuracy',keras.metrics.AUC(name="auc")])

    history = model.fit(X_train, 
                    y_train, 
                    sample_weight = w_train,
                    epochs=MAX_EPOCHS, batch_size=128, 
                    validation_data=(X_val, y_val,w_val), 
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=False)
    
    #we want to maximize the auc (validation set)
    trial_value = max(history.history["val_auc"])

    if trial_value > BEST_VALUE:
        BEST_VALUE = trial_value
        BEST_HISTORY = history
        model.save(f"models/nn_2class{name}.h5")
        model.save(f"models/nn_2class{name}", save_format="tf")
        print("New best model found and saved. Current best value: ", BEST_VALUE)

    return trial_value

BEST_VALUE =  - np.inf
BEST_HISTORY = None

study = optuna.create_study(study_name=f"nn_2class{name}", direction="maximize")



for epochs in [50,100,250,500,1000,1500,2000,2500]:
    MAX_EPOCHS = epochs
    study.optimize(objective, n_trials=N_TRIALS, catch=())

study.trials_dataframe().sort_values(by="value").to_csv(f"models/optuna_trials_nn_2class{name}.csv")

history = BEST_HISTORY

print("Plotting history")
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'models/nn_2class{name}_history_loss.png', transparent=False)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'models/nn_2class{name}_history_acc.png', transparent=False)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['auc'], label='Train Auc')
plt.plot(history.history['val_auc'], label='Val Auc')
plt.title('Training and Validation Auc')
plt.xlabel('Epochs')
plt.ylabel('Auc')
plt.legend()
plt.savefig(f'models/nn_2class{name}_history_auc.png', transparent=False)
plt.show()