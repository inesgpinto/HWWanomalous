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
file_SM = uproot.open("data/SM/Events/run_01/unweighted_events.root")
file_CP_IMPAR = uproot.open("data/CP_IMPAR/Events/run_01/unweighted_events.root")
file_CP_PAR = uproot.open("data/CP_PAR/Events/run_01/unweighted_events.root")



particle_tree_SM = file_SM["LHEF/Particle"]
particle_tree_CP_IMPAR = file_CP_IMPAR["LHEF/Particle"]
particle_tree_CP_PAR = file_CP_PAR["LHEF/Particle"]

# Converting to panda dataframes
print("converting files to df")
SM = particle_tree_SM.arrays(particle_tree_SM.keys(), library="pd")
CP_IMPAR = particle_tree_CP_IMPAR.arrays(particle_tree_CP_IMPAR.keys(), library="pd")
CP_PAR = particle_tree_CP_PAR.arrays(particle_tree_CP_PAR.keys(), library="pd")

print("Variables directly from madgraph:")
print(SM.columns)

#Defining the functions to calculate new variables


def delta_eta_leptons(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-11, -13, -15])]
    lepton2 = df[df['Particle.PID'].isin([11, 13, 15])]
    etas_diff = lepton2['Particle.Eta'].iloc[0] - lepton1['Particle.Eta'].iloc[0]
    return etas_diff

def delta_phi_leptons(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-11, -13, -15])]
    lepton2 = df[df['Particle.PID'].isin([11, 13, 15])]
    phis_diff = lepton2['Particle.Phi'].iloc[0] - lepton1['Particle.Phi'].iloc[0]

    while (phis_diff >= math.pi): 
        phis_diff -= (2*math.pi)
    while (phis_diff < - math.pi): 
        phis_diff += (2*math.pi) 
    
    return phis_diff

def delta_eta_b(df):
    
    b1 = df[df['Particle.PID'].isin([-5])]
    b2 = df[df['Particle.PID'].isin([5])]
    etas_diff = b2['Particle.Eta'].iloc[0] - b1['Particle.Eta'].iloc[0]
    return etas_diff

def delta_phi_b(df):
    
    b1 = df[df['Particle.PID'].isin([-5])]
    b2 = df[df['Particle.PID'].isin([5])]
    phis_diff = b2['Particle.Phi'].iloc[0] - b1['Particle.Phi'].iloc[0]

    while (phis_diff >= math.pi): 
        phis_diff -= (2*math.pi)
    while (phis_diff < - math.pi): 
        phis_diff += (2*math.pi) 
    
    return phis_diff


def delta_eta_neutrinos(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-12, -14, -16])]
    lepton2 = df[df['Particle.PID'].isin([12, 14, 16])]
    etas_diff = lepton2['Particle.Eta'].iloc[0] - lepton1['Particle.Eta'].iloc[0]
    
    return etas_diff

def delta_phi_neutrinos(df):
    """Difference between the two leptons"""
    lepton1 = df[df['Particle.PID'].isin([-12, -14, -16])]
    lepton2 = df[df['Particle.PID'].isin([12, 14, 16])]
    phis_diff = lepton2['Particle.Phi'].iloc[0] - lepton1['Particle.Phi'].iloc[0]
    while (phis_diff >= math.pi): 
        phis_diff -= (2*math.pi)
    while (phis_diff < -  math.pi): 
        phis_diff += (2*math.pi)     
    return phis_diff

def pt_wplus(df):
    wplus = df[df['Particle.PID'] == 24]

    px_w = wplus.iloc[0]['Particle.Px']
    py_w = wplus.iloc[0]['Particle.Py']

    pt_w_plus = np.sqrt(px_w**2 + py_w**2)

    return pt_w_plus

def pt_wminus(df):
    wminus = df[df['Particle.PID'] == -24]

    px_w = wminus.iloc[0]['Particle.Px']
    py_w = wminus.iloc[0]['Particle.Py']

    pt_w_minus = np.sqrt(px_w**2 + py_w**2)

    return pt_w_minus


def pt_WW(df):

    w_bosons = df[df['Particle.PID'].isin([24, -24])]
    
    px_WW = (w_bosons.iloc[0]['Particle.Px'] + w_bosons.iloc[1]['Particle.Px'])
    py_WW = (w_bosons.iloc[0]['Particle.Py'] + w_bosons.iloc[1]['Particle.Py'])
    
    pt_WW = np.sqrt(px_WW**2 + py_WW**2)
    
    return pt_WW


def pt_WWH(df):

    w_bosons = df[df['Particle.PID'].isin([24, -24])]
    higgs = df[df['Particle.PID'] == 25]
    
    px_WWH = (w_bosons.iloc[0]['Particle.Px'] + w_bosons.iloc[1]['Particle.Px'] + higgs.iloc[0]['Particle.Px'])
    py_WWH = (w_bosons.iloc[0]['Particle.Py'] + w_bosons.iloc[1]['Particle.Py'] + higgs.iloc[0]['Particle.Py'])
    
    pt_WWH = np.sqrt(px_WWH**2 + py_WWH**2)
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
    
    m_ww = np.sqrt( E_sum**2 - ( px_sum**2 + py_sum**2 + pz_sum**2 ))
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
    
    m_wwh = np.sqrt(E_sum**2 -(px_sum**2 + py_sum**2 + pz_sum**2 ))
    
    return m_wwh


#Create a dataframe with the new variables

def new_variables(dataframe):
    etas_leptons = []
    phis_leptons = []
    etas_neutrinos = []
    phis_neutrinos = []
    pts_WW = []
    pts_WWH = []
    ms_WW = []
    ms_WWH = []
    pts_wplus = []
    pts_wminus = []
    b_phi = []
    b_eta = []
    
    for entry, new_df in dataframe.groupby(level=0):
        etas_leptons.append(delta_eta_leptons(new_df))
        phis_leptons.append(delta_phi_leptons(new_df))
        etas_neutrinos.append(delta_eta_neutrinos(new_df))
        phis_neutrinos.append(delta_phi_neutrinos(new_df))
        pts_WW.append(pt_WW(new_df))
        pts_WWH.append(pt_WWH(new_df))
        ms_WW.append(m_WW(new_df))
        ms_WWH.append(m_WWH(new_df))
        pts_wplus.append(pt_wplus(new_df))
        pts_wminus.append(pt_wminus(new_df))
        b_phi.append(delta_phi_b(new_df))
        b_eta.append(delta_eta_b(new_df))


    new_variables = pd.DataFrame({'delta_eta_leptons': etas_leptons, 'delta_phi_leptons': phis_leptons,'delta_eta_neutrinos': etas_neutrinos, 'delta_phi_neutrinos': phis_neutrinos, 'pt_WW': pts_WW, 'pt_WWH': pts_WWH, 'm_WW': ms_WW, 'm_WWH': ms_WWH,
                                  'pt_wplus':pts_wplus,'pt_wminus':pts_wminus, 'delta_eta_b':b_eta, 'delta_phi_b': b_phi})
    
    return new_variables


#Defining a new dataframe for each root file
print("creating new variables SM and saving")
new_SM = new_variables(SM)
new_SM.to_hdf("data/sm.h5", key='table',mode='w')
print("creating new variables CP IMPAR and saving")
new_CP_IMPAR = new_variables(CP_IMPAR)
new_CP_IMPAR.to_hdf("data/cp_impar.h5", key='table',mode='w')
print("creating new variables CP PAR and saving")
new_CP_PAR = new_variables(CP_PAR)
new_CP_PAR.to_hdf("data/cp_par.h5", key='table',mode='w')





dataframes = [new_SM, new_CP_IMPAR, new_CP_PAR]
