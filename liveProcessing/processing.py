import numpy as np
from joblib import load

from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import json
import sys
from pymongo.mongo_client import MongoClient
import time

import argparse
import joblib
#datetime
import datetime
import mlflow
from pymongo import MongoClient
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './')
import analysis_functions.btsAnalysis as baf
import analysis_functions.XSensAnalysis as xaf

import pandas as pd


default_logged = 'runs:/a962f0aceae543919833fe9e3fe971ed/'

# default_logged_model_pca = 'runs:/a962f0aceae543919833fe9e3fe971ed/modello_pca'
default_mongodb_uri = 'mongodb://10.250.4.35:27017'
default_mlflow_tracking_uri = 'http://10.250.4.35:8000'
default_mlflow_experiment = 'XSense Training'
MQTTADDRESS = '10.250.4.35'
MQTTPORT = 1883

# Parsing degli argomenti da linea di comando
parser = argparse.ArgumentParser(description='Parametri per il programma')
parser.add_argument('--using_specific_model', default=None, help='Model id')
parser.add_argument('--mqtt_address', default=MQTTADDRESS, help='Indirizzo del server MQTT')
parser.add_argument('--mqtt_port', default=MQTTPORT, help='Porta del server MQTT')
parser.add_argument('--mongodb_uri', default=default_mongodb_uri, help='URI di MongoDB')
parser.add_argument('--mlflow_tracking_uri', default=default_mlflow_tracking_uri, help='URI di tracciamento di MLFlow')
parser.add_argument('--mlflow_experiment', default=default_mlflow_experiment, help='Nome dell\'esperimento di MLFlow')
parser.add_argument('--local', default=False, help='Utilizzare il modello salvato in locale nella cartella modelli?')
args = parser.parse_args()

# Imposta i parametri con i valori forniti o i valori predefiniti
MQTTADDRESS = args.mqtt_address
MQTTPORT = args.mqtt_port  
using_specific_model = args.using_specific_model
mongodb_uri = args.mongodb_uri
mlflow_tracking_uri = args.mlflow_tracking_uri
mlflow_experiment = args.mlflow_experiment
local = args.local

# Connessione a MongoDB
Mongoclient = MongoClient(mongodb_uri)

# Impostazione di MLFlow
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment)


if local:
    PCA = joblib.load('modelli/modelli_corpo/pca.pkl')
    MODEL = load_model('modelli/modelli_corpo/modello.keras')
else:
    if using_specific_model:
        PCA = mlflow.sklearn.load_model(args.using_specific_model, "/modello_pca")
        MODEL = mlflow.pyfunc.load_model(args.using_specific_model, "/modello_keras")
    else:
        # upload_time = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(mlflow_experiment).experiment_id)["run_id"][0]
        PCA = mlflow.sklearn.load_model(f"{default_logged}modello_pca")
        MODEL = mlflow.keras.load_model(f"{default_logged}modello_keras")
        print("Modello caricato con successo")


# print(PCA, MODEL)

# PCA = joblib.load('modelli/pca.pkl')
# MODEL = load_model('modelli/model.keras')

LAST_XSENS_DATA = None
PREDICTIONS = []
VALORESCORSO = ""
NUOVO_VALORE = None
INITIAL_TIMESTAMP = None
EXPERIMENT_LABEL : str | None = None
OBJECTID = None
XSENS_TIMEOUT = 0
MISURECORPO = {}

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

def Xsens_processing(elen):
    global PREDICTIONS
    global MODEL
    global PCA
    global MISURE_CORPO

    frame_number = []
    dfs = []    
    datasetSenzaTagli = []

    # desired_order = ['Joint Angles ZXY_L5S1 Flexion/Extension',
    # 'Joint Angles ZXY_L4L3 Flexion/Extension',
    # 'Joint Angles ZXY_L1T12 Flexion/Extension',
    # 'Joint Angles ZXY_T9T8 Flexion/Extension',
    # 'Joint Angles ZXY_Right Shoulder Internal/External Rotation',
    # 'Joint Angles ZXY_Right Shoulder Flexion/Extension',
    # 'Joint Angles ZXY_Right Elbow Ulnar Deviation/Radial Deviation',
    # 'Joint Angles ZXY_Right Elbow Pronation/Supination',
    # 'Joint Angles ZXY_Right Elbow Flexion/Extension',
    # 'Joint Angles ZXY_Left T4 Shoulder Abduction/Adduction',
    # 'Joint Angles ZXY_Left T4 Shoulder Internal/External Rotation',
    # 'Joint Angles ZXY_Left Shoulder Internal/External Rotation',
    # 'Joint Angles ZXY_Left Shoulder Flexion/Extension',
    # 'Joint Angles ZXY_Left Elbow Pronation/Supination',
    # 'Joint Angles ZXY_Left Elbow Flexion/Extension',
    # 'Segment Position_Right Forearm_x',
    # 'Segment Position_Right Forearm_z',
    # 'Segment Position_Left Upper Arm_z',
    # 'Segment Position_Left Forearm_x',
    # 'Segment Position_Left Forearm_z']

    desired_order = ['Segment Position_Right Forearm_x', 'Segment Position_Right Forearm_z',
       'Segment Position_Left Upper Arm_z', 'Segment Position_Left Forearm_x',
       'Segment Position_Left Forearm_z',
       'Joint Angles ZXY_L5S1 Flexion/Extension',
       'Joint Angles ZXY_L4L3 Flexion/Extension',
       'Joint Angles ZXY_L1T12 Flexion/Extension',
       'Joint Angles ZXY_T9T8 Flexion/Extension',
       'Joint Angles ZXY_Right Shoulder Internal/External Rotation',
       'Joint Angles ZXY_Right Shoulder Flexion/Extension',
       'Joint Angles ZXY_Right Elbow Ulnar Deviation/Radial Deviation',
       'Joint Angles ZXY_Right Elbow Pronation/Supination',
       'Joint Angles ZXY_Right Elbow Flexion/Extension',
       'Joint Angles ZXY_Left T4 Shoulder Abduction/Adduction',
       'Joint Angles ZXY_Left T4 Shoulder Internal/External Rotation',
       'Joint Angles ZXY_Left Shoulder Internal/External Rotation',
       'Joint Angles ZXY_Left Shoulder Flexion/Extension',
       'Joint Angles ZXY_Left Elbow Pronation/Supination',
       'Joint Angles ZXY_Left Elbow Flexion/Extension', 'bodyHeight',
       'footSize', 'shoulderHeight', 'shoulderWidth', 'elbowSpan']
    
    for first in elen:  
        frame_number.append(first['frameNumber'])

        df = readjson(first)

        RMatrix, df = xaf.quaternioni(df)
        df, finaleTers = xaf.ternions(df, RMatrix)
        finaleTers = xaf.xyzSeparazione(finaleTers)

        
        datasetSenzaTagli.append(finaleTers)
        finalissimo = unioneFiles(df, finaleTers)
        #finalissimo add misure corpo
        finalissimo['bodyHeight'] = MISURE_CORPO['bodyHeight']
        finalissimo['footSize'] = MISURE_CORPO['footSize']
        finalissimo['shoulderHeight'] = MISURE_CORPO['shoulderHeight']
        finalissimo['shoulderWidth'] = MISURE_CORPO['shoulderWidth']
        finalissimo['elbowSpan'] = MISURE_CORPO['elbowSpan']
        # print(finalissimo)


        # finalllll = pd.DataFrame(finalissimo, index=[0])
        # print(finalllll.shape)



        # Assuming 'elena' is your dictionary
        elena_reordered = {key: finalissimo[key] for key in desired_order}
        elena_reordered_numpy = np.array(list(elena_reordered.values()))


        dfs.append(elena_reordered_numpy)

    # tempo2 = time.time()
    prediction_result = xaf.predizione(dfs, MODEL, PCA)
    # print("Tempo di esecuzione per la predizione:", time.time() - tempo2, "secondi")
    
    PREDICTIONS.append(prediction_result)
    # tempo2 = time.time()
    prediction_result = [round(num[0]) for num in prediction_result] 

    # finaleTers + prediction_result + frame_number
    perMongoPulito = []
    for i in range(len(datasetSenzaTagli)):
        elemento = {
            "prediction": prediction_result[i],
            "frameNumber": frame_number[i],
            "processedData": datasetSenzaTagli[i]
        }
        perMongoPulito.append(elemento)

    return perMongoPulito

def bts_preprocessing(newSamples, mGClient):
    global INITIAL_TIMESTAMP
    try:
        timek = time.time()

        db = mGClient['experiments']
        collection = db["provaRMS"]
        data, integralVal = baf.obtain_mongo(INITIAL_TIMESTAMP, collection)
        data2 = baf.obtaining_mqtt(newSamples, data)
        lastindex = newSamples[0]["index"]
        data = data2.copy()
        data = baf.filtering(data)
        rms = baf.rms(data)
        integralVal = baf.integralBTS(data, integralVal, lastindex)
        # MMF = baf.meanMedianFreq(data)
        baf.save_mongo(data2, INITIAL_TIMESTAMP, collection,integralVal)

        print(len(data2[1]))
        print("time bts processing: ", time.time()-timek)
        finaleCeriel = []
        for label in rms:
            # Create a dictionary for each sensor label containing rms and MMF values
            sensor_data = {
                "sensorLabel": label,
                "rms": rms[label],
                "effort": integralVal[label]/0.008
            }
            # Append the dictionary to the data list
            finaleCeriel.append(sensor_data)

        # Return the list of dictionaries
        return {"data": finaleCeriel}

    except Exception as e:
        print(e.with_traceback())
        return {"data": []}

def preprocess_data(data, Mongoclient):
    global PREDICTIONS
    global MODEL
    global PCA
    data_xsens = data["xsens_batch"]
    data_bts = data["bts_batch"]

    # Processa i dati Xsens
    data_xsens = Xsens_processing(data_xsens)
    # print(data_xsens)

    # Processa i dati BTS
    data_bts = bts_preprocessing(data_bts, Mongoclient)

    return data_xsens, data_bts

def cambioNomeCollection():
    global Mongoclient
    global EXPERIMENT_LABEL
    db = Mongoclient['experiments']
    # collection = db[f'livedata']
    # collection.rename(f'Exp_{EXPERIMENT_LABEL}_live')
    # print("Collection rinominata")
    #delata collection prova RMS
    collection = db["provaRMS"]
    collection.drop()
    print("Collection provaRMS eliminata")
    
def uploadPredictions(xsens, bts, mGClient, xsens_batch):
    global OBJECTID
    # global INITIAL_TIMESTAMP
    db = mGClient['experiments']   
    collection = db['livedata']
    #salva su txt data
    # with open('data.txt', 'w') as outfile:
    #     json.dump(data, outfile)
    timek = time.time()

    
    xsens_batch.pop("frameNumber")
    xsens_batch.pop("timestamp")
    elemento = {
        "prediction": xsens[-1]["prediction"],
        # "frameNumber": xsens[-1]["frameNumber"],
        "kinematicData": xsens_batch,
        #"Envelope_bts": bts[i],
        # "timestamp": time.time(),
    }
    now = datetime.datetime.now()
    # Formattare la data e l'ora nel formato desiderato
    # Ottenere una stringa formattata in ISO 8601
    iso_format = now.isoformat()

    formatted_datetime = iso_format[:-3] + 'Z'
    msg_data = {
        "experimentId": OBJECTID,
        ""
        "timestamp": formatted_datetime,
        "xsensData": elemento,
        "btsData": bts
    }
    print("Tempo di esecuzione per il preprocessing delle predizioni:", time.time() - timek, "secondi")

    timek = time.time()
    collection.insert_one(msg_data)
    print("Tempo di esecuzione per l'upload delle predizioni:", time.time() - timek, "secondi")

def check_xsens_data_timeout():
    global LAST_XSENS_DATA
    global XSENS_TIMEOUT
    if XSENS_TIMEOUT == 1:
        # Se non hai ancora ricevuto alcun messaggio da xsens, restituisci False
        if LAST_XSENS_DATA is None:
            return False
        

        # Controlla se è trascorso più di 10 secondi dall'ultimo messaggio xsens ricevuto
        if time.time() - LAST_XSENS_DATA > 10:
            print("Timeout: Non ci sono stati nuovi dati xsens per 10 secondi")
            XSENS_TIMEOUT = 0
            return True
        return False
    else:
        return False

def mongoMeta():
    global INITIAL_TIMESTAMP
    global Mongoclient
    
    db = Mongoclient['experiments']  # Replace 'your_database_name' with your actual database name
    # collection = db['your_collection_name']  # Replace 'your_collection_name' with your actual collection name
    # collection or Metadati or XsensSamples or final
    collection = db["metadata"]
    result = None
    while result is None or result["_id"] == "null":
        result = collection.find_one({"experimentStart": f"{INITIAL_TIMESTAMP}"})
        print(result)
        time.sleep(1)

    print(result["_id"])
    print(result)
    return result["_id"]


def on_connect(client, userdata, flags, rc, properties):
    print("Connected with result code "+str(rc))
    print("--------------------------------------------------------------------------------------------------------------------------------------------------")
    client.subscribe("ExperimentInfo",  qos=1)

def on_message(client, userdata, msg):
    global Mongoclient
    global INITIAL_TIMESTAMP
    global EXPERIMENT_LABEL
    global OBJECTID
    global XSENS_TIMEOUT
    global MISURE_CORPO
    if msg.topic == "ExperimentInfo":
        # Avvio dell'esperimento
        mess = json.loads(msg.payload)
        INITIAL_TIMESTAMP = mess["experimentStart"]
        EXPERIMENT_LABEL = mess["experimentLabel"]
        MISURE_CORPO = mess["xsensExperimentData"]['bodyDimensions']

        print("Experiment started.")
        client.subscribe("$share/processing/SamplesData", qos=1)
        OBJECTID = mongoMeta().binary.hex()
        XSENS_TIMEOUT = 1

    elif msg.topic == "SamplesData":
        XSENS_TIMEOUT = 1

        start_time = time.time()

        global VALORESCORSO
        global LAST_XSENS_DATA
        global PREDICTIONS
        # Ricevi un nuovo messaggio da xsens, quindi aggiorna il tempo di ricezione
        LAST_XSENS_DATA = start_time
        global PCA
        global MODEL
        
        try:
            data = msg.payload
            data = data.decode('utf-8')
            data = json.loads(data)
            
            
            

            if VALORESCORSO == msg.payload:
                print("Stesso valore")
                raise KeyboardInterrupt
            else:   
                VALORESCORSO = msg.payload


            
                

            # data_xsens = process_xsens(data)
            # dati_bts = process_bts(data)


            data_xsens, dati_bts = preprocess_data(data, Mongoclient)

            # Carica le previsioni
            uploadPredictions(data_xsens, dati_bts, Mongoclient, data["xsens_batch"][-1])
            
            # print("Tempo di esecuzione per l'upload delle predizioni:", time.time() - tempo2, "secondi")



            print(len(PREDICTIONS))
            
        except KeyboardInterrupt:
            
            cambioNomeCollection()
            print("KeyboardInterrupt, exception riga 731")
            sys.exit()
            


        print("Tempo di esecuzione per la on message:", time.time() - start_time, "secondi")

if __name__ == "__main__":
    
    while True:
        print("Connecting to MQTT broker...")
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTTADDRESS, MQTTPORT)
        LAST_XSENS_DATA = None
        PREDICTIONS = []
        VALORESCORSO = ""
        NUOVO_VALORE = None
        INITIAL_TIMESTAMP = None
        EXPERIMENT_LABEL = None
        OBJECTID = None


        try:
            # You can add additional logic here before starting the loop if needed
            while True:
                if check_xsens_data_timeout():
                    
                    cambioNomeCollection()
                    break
                client.loop()
        except KeyboardInterrupt:
            
            cambioNomeCollection()
            print("KeyboardInterrupt: excepton riga 738")
            sys.exit()