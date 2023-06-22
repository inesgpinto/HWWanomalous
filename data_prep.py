import pandas as pd
import numpy as np
from config import *

print("opening files")
SM = pd.read_hdf("data/sm.h5", key='table',mode='r')
CP_PAR = pd.read_hdf("data/cp_par.h5", key='table',mode='r')
CP_IMPAR = pd.read_hdf("data/cp_impar.h5", key='table',mode='r')

#creating labels
SM["category"]="bkg"
CP_PAR["category"]="signal"
CP_IMPAR["category"]="signal"

SM["sample"]="SM"
CP_PAR["sample"]="CP PAR"
CP_IMPAR["sample"]="CP IMPAR"

if N_CLASSES == 3:
    SM["label"]=0
    CP_PAR["label"]=1
    CP_IMPAR["label"]=2
elif N_CLASSES == 2:
    SM["label"]=0
    CP_PAR["label"]=1
    CP_IMPAR["label"]=1


print("Merging into single dataframe")
appended_data = []
appended_data.append(SM)
appended_data.append(CP_PAR)
appended_data.append(CP_IMPAR)

data = pd.concat(appended_data,ignore_index=True)


print("train/val/test split")
RANDOM_SEED = 42

splits = [ "train", "val", "test"]
data['gen_split']    = ""

indexes_splits = np.split( data.sample(frac=1, random_state=RANDOM_SEED).index,
                [int(1 / 3 * len(data)), int(2 / 3 * len(data))])

for splitname, indexes in zip(splits, indexes_splits):
    data.loc[indexes, "gen_split"] = splitname

print(N_CLASSES)

if N_CLASSES == 3:
    for splitname in splits:
        for sample in data["sample"].unique():
            n_events = len(data.query(f'sample =="{sample}" & gen_split == "{splitname}"'))
            print(f'{splitname} {sample} \t: {n_events} events')

    print("saving data")
    data.to_hdf("data/data_ml.h5", key='table',mode='w')

if N_CLASSES == 2:
    data['train_weight']=1
    for splitname in splits:
        for sample in data["category"].unique():
            n_events = len(data.query(f'category =="{sample}" & gen_split == "{splitname}"'))
            #print(n_events)
            idx = data.query(f'category =="{sample}" & gen_split == "{splitname}"').index

            n_events = len(data.loc[idx])

            data.loc[idx,'train_weight'] = 1./n_events




    for splitname in splits:
        for sample in data["sample"].unique():
            weight_sum = data.query(f'sample =="{sample}" & gen_split == "{splitname}"')['train_weight'].sum()

            print(f'{splitname} {sample} \t: {weight_sum} weight sum')

    print("saving data")
    data.to_hdf("data/data_ml_2class.h5", key='table',mode='w')   
