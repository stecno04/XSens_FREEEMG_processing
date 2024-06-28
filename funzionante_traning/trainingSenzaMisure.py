import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import joblib
import os 
import numpy as np
from tensorflow.keras.utils import plot_model

import re
import mlflow
import mlflow.keras
import ast
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training Xsens classifier without body dimentions.')

parser.add_argument('--mlflow', action=argparse.BooleanOptionalAction, default=False, help='Store the model in MLflow')
parser.add_argument('--local', default=True, action=argparse.BooleanOptionalAction, help='Store the model in local file')
parser.add_argument('--mlflow_uri', type=str, default='http://10.250.4.35:8000', help='MLflow URI')
parser.add_argument('--mlfow_experiment', type=str, default='XSense Training', help='MLflow experiment name')





def readExcel(file_excel):
    xls = pd.ExcelFile(file_excel)
    dfs = []  # Lista per memorizzare i DataFrame di ciascun foglio
    listdrop = ['General Information', 
    'Markers',
    'Segment Velocity',
    'Segment Acceleration',
    'Segment Angular Velocity',
    'Segment Angular Acceleration',
    'Sensor Free Acceleration',
    'Ergonomic Joint Angles ZXY',
    'Ergonomic Joint Angles XZY',
    'Center of Mass',
    'Sensor Magnetic Field',
    'Sensor Orientation - Quat',
    'Ergonomic Joint Angles XZY',
    'Joint Angles XZY',
    'Sensor Orientation - Euler',
    'Segment Orientation - Euler']
    # Leggi ciascun foglio separatamente e aggiungi il nome del foglio come prefisso alle colonne
    for sheet_name in xls.sheet_names:
        if sheet_name in listdrop:
            continue
        df = pd.read_excel(file_excel, sheet_name=sheet_name)
        df.columns = [f"{sheet_name}_{col}" for col in df.columns]  # Rinomina le colonne
        
        if sheet_name == 'Joint Angles ZXY':
            # Calcola la somma di tutti i valori per ogni colonna
            col_sum = df.sum()

            # Seleziona le colonne che hanno somma diversa da zero
            non_zero_cols = col_sum[col_sum != 0].index

            # Seleziona solo le colonne non zero
            df = df[non_zero_cols]

        
        dfs.append(df)  # Aggiungi il DataFrame corrente alla lista
        print(sheet_name)


    # Concatena i DataFrame in un unico DataFrame
    elena = pd.concat(dfs, axis=1)  # Concatenazione in orizzontale
    if 'Stefano' in file_excel:
        # ogni cinque dati prendine 4
        elena = elena.iloc[~(elena.index % 5 == 0)]

    return elena

def dropColumns(elena):
    dropcol = ['Joint Angles ZXY_L5S1 Lateral Bending', 
        'Joint Angles ZXY_L4L3 Lateral Bending', 
        'Joint Angles ZXY_L1T12 Lateral Bending', 
        'Joint Angles ZXY_T9T8 Lateral Bending', 
        'Joint Angles ZXY_T1C7 Axial Rotation', 
        'Joint Angles ZXY_T1C7 Flexion/Extension', 
        'Joint Angles ZXY_C1 Head Axial Rotation', 
        'Joint Angles ZXY_C1 Head Flexion/Extension', 
        'Joint Angles ZXY_Right T4 Shoulder Abduction/Adduction', 
        'Joint Angles ZXY_Right T4 Shoulder Internal/External Rotation', 
        'Joint Angles ZXY_Right T4 Shoulder Flexion/Extension', 
        'Joint Angles ZXY_Right Shoulder Abduction/Adduction', 
        'Joint Angles ZXY_Left Elbow Ulnar Deviation/Radial Deviation',
        'Segment Orientation - Quat_Right Upper Leg q0',
       'Segment Orientation - Quat_L5 q0', 'Segment Orientation - Quat_L5 q1',
       'Segment Orientation - Quat_L5 q2', 'Segment Orientation - Quat_L5 q3',
       'Segment Orientation - Quat_L3 q0', 'Segment Orientation - Quat_L3 q1',
       'Segment Orientation - Quat_L3 q2', 'Segment Orientation - Quat_L3 q3',
       'Segment Orientation - Quat_T12 q0',
       'Segment Orientation - Quat_T12 q1',
       'Segment Orientation - Quat_T12 q2',
       'Segment Orientation - Quat_T12 q3', 'Segment Orientation - Quat_T8 q0',
       'Segment Orientation - Quat_T8 q1', 'Segment Orientation - Quat_T8 q2',
       'Segment Orientation - Quat_T8 q3',
       'Segment Orientation - Quat_Neck q0',
       'Segment Orientation - Quat_Neck q1',
       'Segment Orientation - Quat_Neck q2',
       'Segment Orientation - Quat_Neck q3',
       'Segment Orientation - Quat_Head q0',
       'Segment Orientation - Quat_Head q1',
       'Segment Orientation - Quat_Head q2',
       'Segment Orientation - Quat_Head q3',
       'Segment Orientation - Quat_Right Shoulder q0',
       'Segment Orientation - Quat_Right Shoulder q1',
       'Segment Orientation - Quat_Right Shoulder q2',
       'Segment Orientation - Quat_Right Shoulder q3',
       'Segment Orientation - Quat_Right Upper Arm q0',
       'Segment Orientation - Quat_Right Upper Arm q1',
       'Segment Orientation - Quat_Right Upper Arm q2',
       'Segment Orientation - Quat_Right Upper Arm q3',
       'Segment Orientation - Quat_Right Forearm q0',
       'Segment Orientation - Quat_Right Forearm q1',
       'Segment Orientation - Quat_Right Forearm q2',
       'Segment Orientation - Quat_Right Forearm q3',
       'Segment Orientation - Quat_Right Hand q0',
       'Segment Orientation - Quat_Right Hand q1',
       'Segment Orientation - Quat_Right Hand q2',
       'Segment Orientation - Quat_Right Hand q3',
       'Segment Orientation - Quat_Left Shoulder q0',
       'Segment Orientation - Quat_Left Shoulder q1',
       'Segment Orientation - Quat_Left Shoulder q2',
       'Segment Orientation - Quat_Left Shoulder q3',
       'Segment Orientation - Quat_Left Upper Arm q0',
       'Segment Orientation - Quat_Left Upper Arm q1',
       'Segment Orientation - Quat_Left Upper Arm q2',
       'Segment Orientation - Quat_Left Upper Arm q3',
       'Segment Orientation - Quat_Left Forearm q0',
       'Segment Orientation - Quat_Left Forearm q1',
       'Segment Orientation - Quat_Left Forearm q2',
       'Segment Orientation - Quat_Left Forearm q3', 'Segment Position_Frame',
    'Sensor Orientation - Quat_Left Hand q0',
    'Sensor Orientation - Quat_Left Hand q1',
    'Sensor Orientation - Quat_Left Hand q2',
    'Sensor Orientation - Quat_Left Hand q3',
    'Sensor Orientation - Quat_Right Upper Leg q0',
    'Sensor Orientation - Quat_Right Upper Leg q1',
    'Sensor Orientation - Quat_Right Upper Leg q2',
    'Sensor Orientation - Quat_Right Upper Leg q3',
    'Sensor Orientation - Quat_Right Lower Leg q0',
    'Sensor Orientation - Quat_Right Lower Leg q1',
    'Sensor Orientation - Quat_Right Lower Leg q2',
    'Sensor Orientation - Quat_Right Lower Leg q3',
    'Sensor Orientation - Quat_Right Foot q0',
    'Sensor Orientation - Quat_Right Foot q1',
    'Sensor Orientation - Quat_Right Foot q2',
    'Sensor Orientation - Quat_Right Foot q3',
    'Sensor Orientation - Quat_Right Toe q0',
    'Sensor Orientation - Quat_Right Toe q1',
    'Sensor Orientation - Quat_Right Toe q2',
    'Sensor Orientation - Quat_Right Toe q3',
    'Sensor Orientation - Quat_Left Upper Leg q0',
    'Sensor Orientation - Quat_Left Upper Leg q1',
    'Sensor Orientation - Quat_Left Upper Leg q2',
    'Sensor Orientation - Quat_Left Upper Leg q3',
    'Sensor Orientation - Quat_Left Lower Leg q0',
    'Sensor Orientation - Quat_Left Lower Leg q1',
    'Sensor Orientation - Quat_Left Lower Leg q2',
    'Sensor Orientation - Quat_Left Lower Leg q3',
    'Sensor Orientation - Quat_Left Foot q0',
    'Sensor Orientation - Quat_Left Foot q1',
    'Sensor Orientation - Quat_Left Foot q2',
    'Sensor Orientation - Quat_Left Foot q3',
    'Sensor Orientation - Quat_Left Toe q0',
    'Sensor Orientation - Quat_Left Toe q1',
    'Sensor Orientation - Quat_Left Toe q2',
    'Sensor Orientation - Quat_Left Toe q3',
    'Joint Angles ZXY_Left Wrist Ulnar Deviation/Radial Deviation',
    'Joint Angles ZXY_Left Wrist Pronation/Supination',
    'Joint Angles ZXY_Left Wrist Flexion/Extension',
    'Joint Angles ZXY_Right Wrist Ulnar Deviation/Radial Deviation',
    'Joint Angles ZXY_Right Wrist Pronation/Supination',
    'Joint Angles ZXY_Right Wrist Flexion/Extension',
    'Segment Position_Right Hand x',
    'Segment Position_Right Hand y',
    'Segment Position_Right Hand z',
    'Segment Position_Left Upper Leg x',
    'Segment Position_Left Upper Leg y',
    'Segment Position_Left Upper Leg z',
    'Segment Position_Left Lower Leg x',
    'Segment Position_Left Lower Leg y',
    'Segment Position_Left Lower Leg z',
    'Segment Position_Left Foot x',
    'Segment Position_Left Foot y',
    'Segment Position_Left Foot z',
    'Segment Position_Left Toe x',
    'Segment Position_Left Toe y',
    'Segment Position_Left Toe z',
    'Segment Position_Left Hand x',
    'Segment Position_Left Hand y',
    'Segment Position_Left Hand z',
    'Segment Position_Right Upper Leg x',
    'Segment Position_Right Upper Leg y',
    'Segment Position_Right Upper Leg z',
    'Segment Position_Right Lower Leg x',
    'Segment Position_Right Lower Leg y',
    'Segment Position_Right Lower Leg z',
    'Segment Position_Right Foot x',
    'Segment Position_Right Foot y',
    'Segment Position_Right Foot z',
    'Segment Position_Right Toe x',
    'Segment Position_Right Toe y',
    'Segment Position_Right Toe z',]
    for col in dropcol:
        try:
            elena = elena.drop(columns=col)
        except:
            pass
        
    return elena

def Rotation(q0, q1, q2, q3, x, y, z):
    R_pelv = np.array([[1-(2*(q2**2+q3**2)), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
                        [2 * (q1*q2 + q0*q3), 1-(2*(q1**2+q3**2)), 2 * (q2*q3 - q0*q1)],
                        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1-(2*(q1**2+q2**2))]])

    quart_colonna = np.dot(R_pelv.T, np.array([x, y, z]))

    T = np.eye(4)
    T[:3, 3] = quart_colonna
    T[:3, :3] = R_pelv
    
    return T

def quaternioni(elena):
    # Reset the index of the DataFrame
    elena.reset_index(drop=True, inplace=True)

    # Initialize a new column for the DataFrame
    elena['Segment Orientation - Quat_Pelvis R'] = None

    # Iterate over DataFrame rows using .iterrows()
    for i, row in elena.iterrows():
        q0 = row['Segment Orientation - Quat_Pelvis q0']
        q1 = row['Segment Orientation - Quat_Pelvis q1']
        q2 = row['Segment Orientation - Quat_Pelvis q2']
        q3 = row['Segment Orientation - Quat_Pelvis q3']
        x = row['Segment Position_Pelvis x']
        y = row['Segment Position_Pelvis y']
        z = row['Segment Position_Pelvis z']
        
        # Calculate the rotation matrix
        rotation_matrix = Rotation(q0, q1, q2, q3, x, y, z)
        
        # Assign the rotation matrix to the DataFrame
        elena.at[i, 'Segment Orientation - Quat_Pelvis R'] = rotation_matrix
    return elena

def extract_string(text):
    match = re.search(r'_(.*?)(?=\w$)', text)
    if match:
        return match.group(1).rstrip()
    
def T(R, x, y, z, nome_colonna, index, elena):
    
    matrice = np.array(R)

    # Calcola la trasposta della matrice
    matrice_trasposta = np.transpose(matrice)


    # Definisci il vettore
    vettore = np.array([x, y, z, 1])

    # Moltiplica la trasposta della matrice per il vettore
    risultato = np.dot(matrice_trasposta, vettore)

    # Esempio di utilizzo di DataFrame.loc per creare una colonna se non esiste
    if f'Segment Position_{extract_string(nome_colonna)}' not in elena.columns:
        elena[f'Segment Position_{extract_string(nome_colonna)}'] = None
        elena[f'Segment Position_{extract_string(nome_colonna)}'] = elena[f'Segment Position_{extract_string(nome_colonna)}'].astype(object)
    

    # Convert the array to a string representation
    array_string = str(risultato.tolist())

    # Store the string representation in the DataFrame cell
    elena.loc[index, f'Segment Position_{extract_string(nome_colonna)}'] = array_string

    return elena

def ternions(elena):
    listacolon = [
       'Segment Position_L5 x', 'Segment Position_L5 y', 'Segment Position_Pelvis x',
       'Segment Position_Pelvis y', 'Segment Position_Pelvis z',
       'Segment Position_L5 z', 'Segment Position_L3 x',
       'Segment Position_L3 y', 'Segment Position_L3 z',
       'Segment Position_T12 x', 'Segment Position_T12 y',
       'Segment Position_T12 z', 'Segment Position_T8 x',
       'Segment Position_T8 y', 'Segment Position_T8 z',
       'Segment Position_Neck x', 'Segment Position_Neck y',
       'Segment Position_Neck z', 'Segment Position_Head x',
       'Segment Position_Head y', 'Segment Position_Head z',
       'Segment Position_Right Shoulder x',
       'Segment Position_Right Shoulder y',
       'Segment Position_Right Shoulder z',
       'Segment Position_Right Upper Arm x',
       'Segment Position_Right Upper Arm y',
       'Segment Position_Right Upper Arm z',
       'Segment Position_Right Forearm x', 'Segment Position_Right Forearm y',
       'Segment Position_Right Forearm z', 'Segment Position_Left Shoulder x',
       'Segment Position_Left Shoulder y', 'Segment Position_Left Shoulder z',
       'Segment Position_Left Upper Arm x',
       'Segment Position_Left Upper Arm y',
       'Segment Position_Left Upper Arm z', 'Segment Position_Left Forearm x',
       'Segment Position_Left Forearm y', 'Segment Position_Left Forearm z']

    for i in range(len(elena)):
        Rmat = elena['Segment Orientation - Quat_Pelvis R'][i]
        listsens = []
        for col in elena.columns:
            if col in listacolon:
                match = re.match(r'(.+) ', col)
                if match:
                    substring = match.group(1)
                    if substring not in listsens:
                        listsens.append(substring)
                        x = elena[substring + ' x'][i]
                        y = elena[substring + ' y'][i]
                        z = elena[substring + ' z'][i]
                        elena = T(Rmat, x, y, z, col, i, elena)
    


    daRimuovere = ['Segment Orientation - Quat_Frame','Segment Position_L5',
       'Segment Position_L3', 'Segment Position_T12', 'Segment Position_T8',
       'Segment Position_Neck',
       'Segment Orientation - Quat_Pelvis q0',
       'Segment Orientation - Quat_Pelvis q1',
       'Segment Orientation - Quat_Pelvis q2',
       'Segment Orientation - Quat_Pelvis q3','Segment Orientation - Quat_Frame',
       'Segment Orientation - Quat_Pelvis q0',
       'Segment Orientation - Quat_Pelvis q1',
       'Segment Orientation - Quat_Pelvis q2',
       'Segment Orientation - Quat_Pelvis q3', 'Segment Position_Pelvis x',
       'Segment Position_Pelvis y', 'Segment Position_Pelvis z',
       'Segment Position_L5 x', 'Segment Position_L5 y',
       'Segment Position_L5 z', 'Segment Position_L3 x',
       'Segment Position_L3 y', 'Segment Position_L3 z',
       'Segment Position_T12 x', 'Segment Position_T12 y',
       'Segment Position_T12 z', 'Segment Position_T8 x',
       'Segment Position_T8 y', 'Segment Position_T8 z',
       'Segment Position_Neck x', 'Segment Position_Neck y',
       'Segment Position_Neck z', 'Segment Position_Head x',
       'Segment Position_Head y', 'Segment Position_Head z',
       'Segment Position_Right Shoulder x',
       'Segment Position_Right Shoulder y',
       'Segment Position_Right Shoulder z',
       'Segment Position_Right Upper Arm x',
       'Segment Position_Right Upper Arm y',
       'Segment Position_Right Upper Arm z',
       'Segment Position_Right Forearm x', 'Segment Position_Right Forearm y',
       'Segment Position_Right Forearm z', 'Segment Position_Left Shoulder x',
       'Segment Position_Left Shoulder y', 'Segment Position_Left Shoulder z',
       'Segment Position_Left Upper Arm x',
       'Segment Position_Left Upper Arm y',
       'Segment Position_Left Upper Arm z', 'Segment Position_Left Forearm x',
       'Segment Position_Left Forearm y', 'Segment Position_Left Forearm z',
       'Segment Orientation - Quat_Pelvis R',
       'Segment Orientation - Quat_Left Hand q0',
       'Segment Orientation - Quat_Left Hand q1',
       'Segment Orientation - Quat_Left Hand q2',
       'Segment Orientation - Quat_Left Hand q3',
       'Segment Orientation - Quat_Right Upper Leg q1',	
       'Segment Orientation - Quat_Right Upper Leg q2',	
       'Segment Orientation - Quat_Right Upper Leg q3',	
       'Segment Orientation - Quat_Right Lower Leg q0',	
       'Segment Orientation - Quat_Right Lower Leg q1',	
       'Segment Orientation - Quat_Right Lower Leg q2',	
       'Segment Orientation - Quat_Right Lower Leg q3',	
       'Segment Orientation - Quat_Right Foot q0',	
       'Segment Orientation - Quat_Right Foot q1',	
       'Segment Orientation - Quat_Right Foot q2',	
       'Segment Orientation - Quat_Right Foot q3',	
       'Segment Orientation - Quat_Right Toe q0',	
       'Segment Orientation - Quat_Right Toe q1',	
       'Segment Orientation - Quat_Right Toe q2',	
       'Segment Orientation - Quat_Right Toe q3',	
       'Segment Orientation - Quat_Left Upper Leg q0',	
       'Segment Orientation - Quat_Left Upper Leg q1',	
       'Segment Orientation - Quat_Left Upper Leg q2',	
       'Segment Orientation - Quat_Left Upper Leg q3',	
       'Segment Orientation - Quat_Left Lower Leg q0',	
       'Segment Orientation - Quat_Left Lower Leg q1',	
       'Segment Orientation - Quat_Left Lower Leg q2',	
       'Segment Orientation - Quat_Left Lower Leg q3',	
       'Segment Orientation - Quat_Left Foot q0',	
       'Segment Orientation - Quat_Left Foot q1',	
       'Segment Orientation - Quat_Left Foot q2',	
       'Segment Orientation - Quat_Left Foot q3',	
       'Segment Orientation - Quat_Left Toe q0',	
       'Segment Orientation - Quat_Left Toe q2',
       'Segment Orientation - Quat_Left Toe q1',
       'Segment Orientation - Quat_Left Toe q3',
       'Joint Angles ZXY_Frame']

    for col in elena.columns:
        if col in daRimuovere:
            elena = elena.drop(columns=col)
    return elena

def xyzSeparazione(elena):
    colonnePassate = ['Segment Position_Pelvis',	
              'Segment Position_Head',
              'Segment Position_Right Shoulder',	
              'Segment Position_Right Upper Arm',	
              'Segment Position_Right Forearm',	
              'Segment Position_Left Shoulder',	
              'Segment Position_Left Upper Arm',	
              'Segment Position_Left Forearm',]


    for col in colonnePassate:
        if not f"{col}_x" in elena.columns:
            elena[f"{col}_x"] = None
        if not f"{col}_y" in elena.columns:
            elena[f"{col}_y"] = None
        if not f"{col}_z" in elena.columns:
            elena[f"{col}_z"] = None



    for col in colonnePassate:
        for i in range(len(df)):
            # Convert the scalar value to a string before indexing
            value_str = df[col][i]

            # Parse the string representation back to a list or array
            array_list = ast.literal_eval(value_str)

            
            # Access the first character of the string and assign it to the new column
            df.loc[i, f"{col}_x"] = array_list[0]
            df.loc[i, f"{col}_y"] = array_list[1]
            df.loc[i, f"{col}_z"] = array_list[2]


    # Rimuoviamo le colonne originali
    elena.drop(columns=colonnePassate, inplace=True)
    elena.drop(columns=[
    'Segment Position_Pelvis_x',
    'Segment Position_Pelvis_y',
    'Segment Position_Pelvis_z',
    'Segment Position_Head_y',
    'Segment Position_Head_z',
    'Segment Position_Right Shoulder_x',
    'Segment Position_Right Shoulder_y',
    'Segment Position_Right Shoulder_z',
    'Segment Position_Right Upper Arm_y',
    'Segment Position_Right Upper Arm_z',
    'Segment Position_Right Forearm_y',
    'Segment Position_Left Shoulder_y',
    'Segment Position_Left Shoulder_z',
    'Segment Position_Left Upper Arm_y',
    'Segment Position_Left Forearm_y','Segment Position_Left Shoulder_x',
'Segment Position_Left Upper Arm_x', 'Segment Position_Head_x', 'Joint Angles ZXY_Left Shoulder Abduction/Adduction',
'Joint Angles ZXY_T9T8 Axial Rotation', 'Joint Angles ZXY_L1T12 Axial Rotation',
'Joint Angles ZXY_L4L3 Axial Rotation', 'Joint Angles ZXY_L5S1 Axial Bending',
'Joint Angles ZXY_T1C7 Lateral Bending', 'Joint Angles ZXY_C1 Head Lateral Bending',
'Joint Angles ZXY_Left T4 Shoulder Flexion/Extension', 'Segment Position_Right Upper Arm_x',

], inplace=True)
    return elena

def training(elena):
    elena['Target'] =0
    elena = elena.drop(elena.index[800:1600])
    elena = elena.drop(elena.index[3200:4000])
    elena = elena.drop(elena.index[5600:6400])
    elena = elena.drop(elena.index[8000:8800])
    elena.loc[1600:3200, 'Target'] = 1
    elena.loc[6400:8000, 'Target'] = 1
    elena = elena.drop(elena.index[800+9690:1600+9690])
    elena = elena.drop(elena.index[3200+9690:4000+9690])
    elena = elena.drop(elena.index[5600+9690:6400+9690])
    elena = elena.drop(elena.index[8000+9690:8800+9690])
    elena.loc[1600+9690:3200+9690, 'Target'] = 1
    elena.loc[6400+9690:8000+9690, 'Target'] = 1
    elena = elena.drop(elena.index[6800+19332:7600+19332])
    elena = elena.drop(elena.index[2000+19332:2800+19332])
    elena.loc[2800+19332:6800+19332, 'Target'] = 1
    elena = elena.drop(elena.index[800+29097:1600+29097])
    elena = elena.drop(elena.index[3200+29097:4000+29097])
    elena = elena.drop(elena.index[5600+29097:6400+29097])
    elena = elena.drop(elena.index[8000+29097:8800+29097])
    elena.loc[1600+29097:3200+29097, 'Target'] = 1
    elena.loc[6400+29097:8000+29097, 'Target'] = 1
    elena = elena.drop(elena.index[800+38486:1600+38486])
    elena = elena.drop(elena.index[3200+38486:4000+38486])
    elena = elena.drop(elena.index[5600+38486:6400+38486])
    elena = elena.drop(elena.index[8000+38486:8800+38486])
    elena.loc[1600+38486:3200+38486, 'Target'] = 1
    elena.loc[6400+38486:8000+38486, 'Target'] = 1
    elena = elena.drop(elena.index[2000+48122:2800+48122])
    elena = elena.drop(elena.index[6800+48122:7600+48122])
    elena.loc[2800+48122:6800+48122, 'Target'] = 1
    
    pca = PCA()
    testa = elena['Target']
    traina = elena.drop(['Target'], axis=1)

    traina = traina.to_numpy()
    pca.fit(traina)
    dataset = pca.transform(traina)[:, :20]  # Ridimensionamento a 20 features per semplicità

    # Dividi il dataset in features (X) e target (y)
    X_train = dataset  # Assume che la colonna del target si chiami 'Target'
    y_train = testa

    # Creazione del modello Keras CNN 1D
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu')) 
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1, mode='min')]

    # Compilazione del modello
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Addestramento del modello
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, shuffle=True, callbacks=callback)

    global MLFOW_STORING
    global LOCAL_STORING

    if MLFOW_STORING:
        try:
            # Memorizza il modello Keras utilizzando MLflow
            with mlflow.start_run() as run:
                # Log degli iperparametri
                mlflow.log_param("epochs", 1000)
                mlflow.log_param("batch_size", 32)
                mlflow.log_param("validation_split", 0.2)


                
                mlflow.sklearn.log_model(pca, "modello_pca")
                # Log delle metriche del modello
                mlflow.keras.log_model(model, "modello_keras")
        except Exception as e:
            print(f"Errore durante il salvataggio su MLflow, salvataggio in locale: {e}")
            # Memorizza il modello PCA
            joblib.dump(pca, 'modelli/pca.pkl')
            # Save the Keras model
            model.save('modelli/model.keras')
    if LOCAL_STORING:
        # Memorizza il modello PCA
        joblib.dump(pca, 'modelli/pca.pkl')
        # Save the Keras model
        model.save('modelli/model.keras')


    print("Training completato con successo!")


if __name__ == "__main__":
    # args = parser.parse_args()
    # mlflow.set_tracking_uri(args.mlflow_uri)
    # mlflow.set_experiment(args.experiment_name)
    # MLFOW_STORING = args.mlflow
    # LOCAL_STORING = args.local
    # print(args)
    # Load the data
    # listaFiles = os.listdir('data')
    # print(listaFiles)
    # dfs = []  # Lista per memorizzare i DataFrame di ciascun file
    # lunghezza = [0]
    # for file in listaFiles:
    #     if file.endswith('.xlsx'):
    #         print(file)
    #         df = readExcel('data/' + file)
    #         lunghezza.append(len(df)+lunghezza[-1])
    #         dfs.append(df)
            
    #         # Concatena i DataFrame verticalmente
    # df = pd.concat(dfs, ignore_index=True)



    # print(lunghezza)

    # print("Hello from preprocessing.py")

    # df = dropColumns(df)
    # df = quaternioni(df)
    # df = ternions(df)
    # df = xyzSeparazione(df)

    #save
    # df.to_csv('training.csv', index = False) 

    df = pd.read_csv('./data/training.csv')

    training(df)


