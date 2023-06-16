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
import awkward as ak
import pandas as pd

# Reading the trees

file_SM = uproot.open("data/SM/Events/run_01/unweighted_events.root")
file_CP_IMPAR = uproot.open("data/CP_IMPAR/Events/run_01/unweighted_events.root")
file_CP_PAR = uproot.open("data/CP_PAR/Events/run_01/unweighted_events.root")

particle_tree_SM = file_SM["LHEF/Particle"]
particle_tree_CP_IMPAR = file_CP_IMPAR["LHEF/Particle"]
particle_tree_CP_PAR = file_CP_PAR["LHEF/Particle"]

# Converting to panda dataframes

SM = particle_tree_SM.arrays(particle_tree_SM.keys(), library="pd")
CP_IMPAR = particle_tree_CP_IMPAR.arrays(particle_tree_CP_IMPAR.keys(), library="pd")
CP_PAR = particle_tree_CP_PAR.arrays(particle_tree_CP_PAR.keys(), library="pd")

#Defining the functions to calculate new variables


def delta_eta(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-11, -13, -15])]
    lepton2 = df[df['Particle.PID'].isin([11, 13, 15])]
    etas_diff = lepton2['Particle.Eta'].iloc[0] - lepton1['Particle.Eta'].iloc[0]
    
    return etas_diff

def delta_phi(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-11, -13, -15])]
    lepton2 = df[df['Particle.PID'].isin([11, 13, 15])]
    phis_diff = lepton2['Particle.Phi'].iloc[0] - lepton1['Particle.Phi'].iloc[0]
    
    return phis_diff

def pt_WW(df):

    w_bosons = df[df['Particle.PID'].isin([24, -24])]
    pt_WW = (w_bosons.iloc[0]['Particle.PT'] + w_bosons.iloc[1]['Particle.PT'])
    
    return pt_WW

def pt_WWH(df):

    w_bosons = df[df['Particle.PID'].isin([24, -24])]
    higgs = df[df['Particle.PID'] == 25]
    
    pt_WWH = (w_bosons.iloc[0]['Particle.PT'] + w_bosons.iloc[1]['Particle.PT'] + higgs.iloc[0]['Particle.PT'])
    
    return pt_WWH


def m_WW(df):
    """Calculate invariant mass of two W bosons"""
    w_bosons = df[df['Particle.PID'].isin([24, -24])]
    w1 = w_bosons.iloc[0]  # First W boson
    w2 = w_bosons.iloc[1]  # Second W boson
    px_sum = w1['Particle.Px'] + w2['Particle.Px']
    py_sum = w1['Particle.Py'] + w2['Particle.Py']
    pz_sum = w1['Particle.Pz'] + w2['Particle.Pz']
    E_sum = w1['Particle.E'] + w2['Particle.E']
    
    m_ww = np.sqrt(px_sum**2 + py_sum**2 + pz_sum**2 + E_sum**2)
    return m_ww

def m_WWH(df):
    """Calculate invariant mass of two W bosons and an additional Higgs boson"""
    w_bosons_higgs = df[df['Particle.PID'].isin([24, -24, 25])]
    w1 = w_bosons_higgs.iloc[0] 
    w2 = w_bosons_higgs.iloc[1]
    higgs = w_bosons_higgs.iloc[2]
    px_sum = w1['Particle.Px'] + w2['Particle.Px'] + higgs['Particle.Px']
    py_sum = w1['Particle.Py'] + w2['Particle.Py'] + higgs['Particle.Py']
    pz_sum = w1['Particle.Pz'] + w2['Particle.Pz'] +  higgs['Particle.Pz']
    E_sum = w1['Particle.E'] + w2['Particle.E'] + higgs['Particle.E']
    
    m_wwh = np.sqrt(px_sum**2 + py_sum**2 + pz_sum**2 + E_sum**2)
    
    return m_wwh


#Create a dataframe with the new variables

def new_variables(dataframe):
    etas = []
    phis = []
    pts_WW = []
    pts_WWH = []
    ms_WW = []
    ms_WWH = []
    
    for entry, new_df in dataframe.groupby(level=0):
        etas.append(delta_eta(new_df))
        phis.append(delta_phi(new_df))
        pts_WW.append(pt_WW(new_df))
        pts_WWH.append(pt_WWH(new_df))
        ms_WW.append(m_WW(new_df))
        ms_WWH.append(m_WWH(new_df))

    new_variables = pd.DataFrame({'delta_eta': etas, 'delta_phi': phis, 'pt_WW': pts_WW, 'pt_WWH': pts_WWH, 'm_WW': ms_WW, 'm_WWH': ms_WWH})
    
    return new_variables


#Defining a new dataframe for each root file

new_SM = new_variables(SM)
new_CP_IMPAR = new_variables(CP_IMPAR)
new_CP_PAR = new_variables(CP_PAR)


def plot_histograms(dataframes, variable_name):
    plt.figure(figsize=(8, 6))
    colors = ['steelblue', 'purple', 'green']
    labels = ['SM', 'CP_odd', 'CP_even']

    for i, dataframe in enumerate(dataframes):
        plt.hist(dataframe[variable_name], bins=50, histtype='step', density=True, color=colors[i], lw=1.5, label=labels[i])

    plt.xlabel(variable_name)
    plt.legend()
    plt.savefig(f'{variable_name}_histogram.png', transparent=False)
    plt.close()


dataframes = [new_SM, new_CP_IMPAR, new_CP_PAR]


for variable in new_SM.columns:
    plot_histograms(dataframes, variable)

