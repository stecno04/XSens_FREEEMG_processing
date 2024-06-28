
import sys
from tensorflow.keras.utils import plot_model


sys.path.insert(1, './')
import analysis_functions.XSensAnalysis as xaf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from joblib import load

from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mlflow
import mlflow.sklearn
import mlflow.keras
from pymongo import MongoClient

# caution: path[0] is reserved for script path (or '' in REPL)



default_logged_model = 'runs:/17a2da7e7d3348319edb0a9a669927ee/modello_keras'

default_logged_model_pca = 'runs:/17a2da7e7d3348319edb0a9a669927ee/modello_pca'
default_mongodb_uri = 'mongodb://10.250.4.35:27017'


mlflow_tracking_uri = 'http://10.250.4.35:8000'
mlflow_experiment = 'XSense Training'
# mlflow.set_tracking_uri(mlflow_tracking_uri)
# mlflow.set_experiment(mlflow_experiment)



def readjson(first):
    df = {}
    df['Segment Orientation - Quat_Frame'] = first['frameNumber']
    df['Segment Orientation - Quat_Pelvis q0'] = first['segments']["0"]["orientation"]["q1"]
    df['Segment Orientation - Quat_Pelvis q1'] = first['segments']["0"]["orientation"]["q2"]
    df['Segment Orientation - Quat_Pelvis q2'] = first['segments']["0"]["orientation"]["q3"]
    df['Segment Orientation - Quat_Pelvis q3'] = first['segments']["0"]["orientation"]["q4"]
    df['Segment Position_Pelvis x'] = first['segments']["0"]["position"]["x"]
    df['Segment Position_Pelvis y'] = first['segments']["0"]["position"]["y"]
    df['Segment Position_Pelvis z'] = first['segments']["0"]["position"]["z"]
    df['Segment Position_L5 x'] = first['segments']["1"]["position"]["x"]
    df['Segment Position_L5 y'] = first['segments']["1"]["position"]["y"]
    df['Segment Position_L5 z'] = first['segments']["1"]["position"]["z"]
    df['Segment Position_L3 x'] = first['segments']["2"]["position"]["x"]
    df['Segment Position_L3 y'] = first['segments']["2"]["position"]["y"]
    df['Segment Position_L3 z'] = first['segments']["2"]["position"]["z"]
    df['Segment Position_T12 x'] = first['segments']["3"]["position"]["x"]
    df['Segment Position_T12 y'] = first['segments']["3"]["position"]["y"]
    df['Segment Position_T12 z'] = first['segments']["3"]["position"]["z"]
    df['Segment Position_T8 x'] = first['segments']["4"]["position"]["x"]
    df['Segment Position_T8 y'] = first['segments']["4"]["position"]["y"]
    df['Segment Position_T8 z'] = first['segments']["4"]["position"]["z"]
    df['Segment Position_Neck x'] = first['segments']["5"]["position"]["x"]
    df['Segment Position_Neck y'] = first['segments']["5"]["position"]["y"]
    df['Segment Position_Neck z'] = first['segments']["5"]["position"]["z"]
    df['Segment Position_Head x'] =     first['segments']["6"]["position"]["x"]
    df['Segment Position_Head y'] = first['segments']["6"]["position"]["y"]
    df['Segment Position_Head z'] = first['segments']["6"]["position"]["z"]
    df['Segment Position_Right Shoulder x'] = first['segments']["7"]["position"]["x"]
    df['Segment Position_Right Shoulder y'] = first['segments']["7"]["position"]["y"]
    df['Segment Position_Right Shoulder z'] = first['segments']["7"]["position"]["z"]
    df['Segment Position_Right Upper Arm x'] = first['segments']["8"]["position"]["x"]
    df['Segment Position_Right Upper Arm y'] = first['segments']["8"]["position"]["y"]
    df['Segment Position_Right Upper Arm z'] = first['segments']["8"]["position"]["z"]
    df['Segment Position_Right Forearm x'] = first['segments']["9"]["position"]["x"]
    df['Segment Position_Right Forearm y'] = first['segments']["9"]["position"]["y"]
    df['Segment Position_Right Forearm z'] = first['segments']["9"]["position"]["z"]
    df['Segment Position_Left Shoulder x'] = first['segments']["11"]["position"]["x"]
    df['Segment Position_Left Shoulder y'] = first['segments']["11"]["position"]["y"]
    df['Segment Position_Left Shoulder z'] = first['segments']["11"]["position"]["z"]
    df['Segment Position_Left Upper Arm x'] = first['segments']["12"]["position"]["x"]
    df['Segment Position_Left Upper Arm y'] = first['segments']["12"]["position"]["y"]  
    df['Segment Position_Left Upper Arm z'] = first['segments']["12"]["position"]["z"]
    df['Segment Position_Left Forearm x'] = first['segments']["13"]["position"]["x"]
    df['Segment Position_Left Forearm y'] = first['segments']["13"]["position"]["y"]
    df['Segment Position_Left Forearm z'] = first['segments']["13"]["position"]["z"]

    df['Joint Angles ZXY_L5S1 Axial Bending'] = first['joints']["0-1"]["y"]
    df['Joint Angles ZXY_L5S1 Flexion/Extension'] = first['joints']["0-1"]["z"]
    df['Joint Angles ZXY_L4L3 Axial Rotation'] = first['joints']["1-2"]["y"]
    df['Joint Angles ZXY_L4L3 Flexion/Extension'] = first['joints']["1-2"]["z"]
    df['Joint Angles ZXY_L1T12 Axial Rotation'] = first['joints']["2-3"]["y"]
    df['Joint Angles ZXY_L1T12 Flexion/Extension'] = first['joints']["2-3"]["z"]
    df['Joint Angles ZXY_T9T8 Axial Rotation'] = first['joints']["3-4"]["y"]
    df['Joint Angles ZXY_T9T8 Flexion/Extension'] = first['joints']["3-4"]["z"]
    df['Joint Angles ZXY_T1C7 Lateral Bending'] = first['joints']["4-5"]["x"]
    df['Joint Angles ZXY_C1 Head Lateral Bending'] = first['joints']["5-6"]["x"]
    df['Joint Angles ZXY_Right Shoulder Internal/External Rotation'] = first['joints']["7-8"]["y"]
    df['Joint Angles ZXY_Right Shoulder Flexion/Extension'] = first['joints']["7-8"]["z"]
    df['Joint Angles ZXY_Right Elbow Ulnar Deviation/Radial Deviation'] = first['joints']["8-9"]["x"]
    df['Joint Angles ZXY_Right Elbow Pronation/Supination'] = first['joints']["8-9"]["y"]
    df['Joint Angles ZXY_Right Elbow Flexion/Extension'] = first['joints']["8-9"]["z"]
    df['Joint Angles ZXY_Left T4 Shoulder Abduction/Adduction'] = first['joints']["4-11"]["x"]
    df['Joint Angles ZXY_Left T4 Shoulder Internal/External Rotation'] = first['joints']["4-11"]["y"]
    df['Joint Angles ZXY_Left T4 Shoulder Flexion/Extension'] = first['joints']["4-11"]["z"]
    df['Joint Angles ZXY_Left Shoulder Abduction/Adduction'] = first['joints']["11-12"]["x"]
    df['Joint Angles ZXY_Left Shoulder Internal/External Rotation'] = first['joints']["11-12"]["y"]
    df['Joint Angles ZXY_Left Shoulder Flexion/Extension'] = first['joints']["11-12"]["z"]
    df['Joint Angles ZXY_Left Elbow Pronation/Supination'] = first['joints']["12-13"]["y"]
    df['Joint Angles ZXY_Left Elbow Flexion/Extension'] = first['joints']["12-13"]["z"]
    
    return df

def unioneFiles(elena, final):
    # Cicla su tutte le chiavi del dizionario originale
    for chiave, valore in elena.items():
        # Controlla se la parola 'angels' è contenuta nella chiave
        if 'Angles' in chiave:
            # Aggiunge la chiave e il valore al dizionario risultante
            final[chiave] = valore
    colonnePassate = ['Segment Position_Head_y','Segment Position_Head_z','Segment Position_Right Shoulder_x','Segment Position_Right Shoulder_y','Segment Position_Right Shoulder_z','Segment Position_Right Upper Arm_y','Segment Position_Right Upper Arm_z','Segment Position_Right Forearm_y','Segment Position_Left Shoulder_y','Segment Position_Left Shoulder_z','Segment Position_Left Upper Arm_y','Segment Position_Left Forearm_y','Segment Position_Left Shoulder_x','Segment Position_Left Upper Arm_x','Segment Position_Head_x','Joint Angles ZXY_Left Shoulder Abduction/Adduction','Joint Angles ZXY_T9T8 Axial Rotation','Joint Angles ZXY_L1T12 Axial Rotation','Joint Angles ZXY_L4L3 Axial Rotation','Joint Angles ZXY_L5S1 Axial Bending','Joint Angles ZXY_T1C7 Lateral Bending','Joint Angles ZXY_C1 Head Lateral Bending','Joint Angles ZXY_Left T4 Shoulder Flexion/Extension','Segment Position_Right Upper Arm_x','Segment Position_L3_x','Segment Position_L3_y','Segment Position_L3_z','Segment Position_L5_x','Segment Position_L5_y','Segment Position_L5_z','Segment Position_Neck_x','Segment Position_Neck_y','Segment Position_Neck_z','Segment Position_T12_x','Segment Position_T12_y','Segment Position_T12_z','Segment Position_T8_x','Segment Position_T8_y','Segment Position_T8_z']

    # Rimuovi le colonne originali dal dizionario
    for col in colonnePassate:
        try:
            final.pop(col)
        except:
            pass
    return final


# print("Ciao------------------------------------------------------------------------------------")

def creazione_training(datas, misure, datasetSenzaTagli = []):
    dfs = pd.DataFrame()
    for first in datas:  
        df = readjson(first)
        RMatrix, df = xaf.quaternioni(df)
        df, finaleTers = xaf.ternions(df, RMatrix)
        finaleTers = xaf.xyzSeparazione(finaleTers)
        
        datasetSenzaTagli.append(finaleTers)
        finalissimo = unioneFiles(df, finaleTers)
        finalllll = pd.DataFrame(finalissimo, index=[0])
        dfs = pd.concat([dfs, finalllll]).reset_index(drop=True)
    # su tutte le righe di dfs aggiungi i dati di misure
    for key, value in misure.items():
        dfs[key] =value
        
    return dfs

def getData(default_mongodb_uri):
    mGClient = MongoClient(default_mongodb_uri)

    listaEsperimenti = ["Exp_prova_christian_good_xsens", "Exp_omar_good_oven_xsens", "Exp_omar_acceptable_xsens", "Exp_omar_acceptable_oven_xsens", "Exp_andrea_acceptable_oven_xsens", "Exp_andrea_good_oven_xsens", "Exp_andrea_acceptable_xsens"]
    misure = {"christian": {"bodyHeight": 1.813, "footSize": 0.290, "shoulderHeight": 1.525, "shoulderWidth": 0.335, "elbowSpan": 0.991}, "omar": {"bodyHeight": 1.708, "footSize": 0.308, "shoulderHeight": 1.404, "shoulderWidth": 0.365, "elbowSpan": 0.835}, "andrea": {"bodyHeight": 1.811, "footSize": 0.304, "shoulderHeight": 1.502, "shoulderWidth": 0.410, "elbowSpan": 0.875}}
    personeCompletes = pd.DataFrame()
    for exp in listaEsperimenti:
        print(exp)
        
        collection = mGClient['experiments'][exp]
        datas = collection.find({})
        for x in misure:
            if x in exp:
                persona = x
        personaCompleta = creazione_training(datas, misure[persona])
        personeCompletes = pd.concat([personeCompletes, personaCompleta]).reset_index(drop=True)
    return personeCompletes 

def training(personeCompletes):
    personeCompletes["Target"] = 0
    personeCompletes.loc[0+3000:3000+6000, 'Target'] = 1    #2minuti

    personeCompletes.loc[12000+3000:15000+6000, 'Target'] = 1   #2minuti
    personeCompletes.loc[24000+3000:27000+6000, 'Target'] = 1   #2minuti
    personeCompletes.loc[36000+3000:39000+12000, 'Target'] = 1    #3minuti
    personeCompletes.loc[54000+3000:57000+12000, 'Target'] = 1   #3minuti
    personeCompletes.loc[72000+3000:75000+6000, 'Target'] = 1   #2minuti
    personeCompletes.loc[84000+3000:87000+6000, 'Target'] = 1  #2minuti

    
    # Split del dataset in training e test set
    split = train_test_split(personeCompletes, test_size=0.2, random_state=42)

    personeCompletes = split[0]
    test = split[1]

    # Perform PCA
    pca = PCA()
    testa = personeCompletes['Target']
    traina = personeCompletes.drop(['Target'], axis=1)
    traina = traina.to_numpy()
    pca.fit(traina)
    dataset = pca.transform(traina)[:, :20]  # Ridimensionamento a 20 features per semplicità

    # Dividi il dataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(dataset, testa, test_size=0.2, random_state=42)

    # Reshape data for Conv1D (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Creazione del modello Keras CNN
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(20, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1, mode='min')]

    # Compilazione del modello
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Addestramento del modello
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, shuffle=True, callbacks=callback)


    # Valutazione del modello sul set di test
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Visualizzazione dei grafici di loss e accuracy
    plt.figure(figsize=(14, 6))

    # Grafico della perdita (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Grafico della precisione (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    print(model.summary())


    # Plot the model
    plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)

    import joblib
    pca_filename = 'modelli/modelli_corpo/pca.pkl'
    joblib.dump(pca, pca_filename)
    model.save('modelli/modelli_corpo/modello.keras')



    # with mlflow.start_run() as run:
    #     # Log degli iperparametri
    #     mlflow.log_param("epochs", 1000)
    #     mlflow.log_param("batch_size", 32)
    #     mlflow.log_param("validation_split", 0.2)
        
    #     mlflow.sklearn.log_model(pca, "modello_pca")
    #     # Log delle metriche del modello
    #     mlflow.keras.log_model(model, "modello_keras")


def main():

    # personeCompletes = getData(default_mongodb_uri)
    # personeCompletes.to_csv('experimentsData/dataset_corpo.csv', index=False)
    personeCompletes = pd.read_csv('experimentsData/dataset_corpo.csv')
    training(personeCompletes)
    
if __name__ == '__main__':
    main()