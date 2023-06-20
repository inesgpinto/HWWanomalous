# Particle.PID

# 22 - fotão
# 24 - W+
# 25 - Higgs
# 11 - eletrão
# 12 - neutrino e
# 13 - mu
# 14 - neutrino mu 
# 5 - quark b
# 15 - tau 
# 16 - neutrino tau


#Importing the modules

import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# Reading the trees
print("opening files")

SM = pd.read_hdf("data/sm.h5", key='table',mode='r')
CP_PAR = pd.read_hdf("data/cp_par.h5", key='table',mode='r')
CP_IMPAR = pd.read_hdf("data/cp_impar.h5", key='table',mode='r')


def plot_histograms(dataframes, variable_name):
    plt.figure(figsize=(8, 6))

    bin = 50 
    
    bins = plt.hist(dataframes[0][variable_name], bins=bin, histtype='step', density=True, color='steelblue', lw=1.5, label='SM')
    plt.hist(dataframes[1][variable_name], bins=bins[1], histtype='step', density=True, color='purple', lw=1.5, label='CP_odd')
    plt.hist(dataframes[2][variable_name], bins=bins[1], histtype='step', density=True, color='green', lw=1.5, label='CP_even')


    plt.xlabel(variable_name)
    plt.legend()
    plt.savefig(f'plots/{variable_name}_histogram.png', transparent=False)
    plt.close()


dataframes = [SM, CP_IMPAR, CP_PAR]


for variable in SM.columns:
    print(variable)
    plot_histograms(dataframes, variable)