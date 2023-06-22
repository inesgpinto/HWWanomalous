import matplotlib.pyplot as plt
from config import *
import pandas as pd

print("Reading file")
data = pd.read_hdf("data/data_ml_2class.h5", key='table',mode='r')

for feat in TRAIN_FEATURES:
    print(feat)
    plt.clf()
    fig, ax = plt.subplots()
    plt.hist2d(
        x=data.query("category == 'bkg'")[feat].values,
        y=data.query("sample == 'CP IMPAR'")[feat].values,
        bins=50,
        density = True
        #cmin=0
        )
    plt.title(feat)
    plt.xlabel("bkg")
    plt.ylabel("CP ODD")
    plt.colorbar()
    plt.savefig(f"hists2d/{feat}.png")


for feat1 in TRAIN_FEATURES:
    for feat2 in TRAIN_FEATURES:
        if feat1 == feat2: break
        plt.clf()
        #fig, ax = plt.subplots()
        plt.scatter(
            x=data.query("category == 'bkg'")[feat1].values,
            y=data.query("category == 'bkg'")[feat2].values,
            alpha=0.2
        )
        plt.scatter(
            x=data.query("sample == 'CP IMPAR'")[feat1].values,
            y=data.query("sample == 'CP IMPAR'")[feat2].values,
            alpha=0.2
        )
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.savefig(f"hists2d/scatter_{feat1}_{feat2}.png")