import paho.mqtt.client as mqtt
import json
import sys
from pymongo.mongo_client import MongoClient
import time
import argparse 

MQTTADDRESS = '10.250.4.35'
MQTTPORT = 1883
default_mongodb_uri = 'mongodb://10.250.4.35:27017'

parser = argparse.ArgumentParser(description='Parametri per il programma')
parser.add_argument('--mqtt_address', default=MQTTADDRESS, help='Indirizzo del server MQTT')
parser.add_argument('--mqtt_port', default=MQTTPORT, help='Porta del server MQTT')
parser.add_argument('--mongodb_uri', default=default_mongodb_uri, help='URI di MongoDB')
args = parser.parse_args()



# Imposta i parametri con i valori forniti o i valori predefiniti
MQTTADDRESS = args.mqtt_address
MQTTPORT = args.mqtt_port  
mongodb_uri = args.mongodb_uri
Mongoclient = MongoClient(mongodb_uri)



VALORESCORSO = ""
PREDICTIONS = []
NUOVO_VALORE = None
EXPERIMENTLABEL = None
XSENS_TIMEOUT = 0
LAST_XSENS_DATA = None



def mongoMeta(data, mGClient):
    global INITIAL_TIMESTAMP
    time.sleep(1)
    db = mGClient['experiments']  # Replace 'your_database_name' with your actual database name
    # collection = db['your_collection_name']  # Replace 'your_collection_name' with your actual collection name
    # collection or Metadati or XsensSamples or final
    collection = db["metadata"]
    print(data)
    data["threshold"] = 0.5
    result = collection.insert_one(data)
    print(f'One post: {result.inserted_id}')
    INITIAL_TIMESTAMP = data["experimentStart"]
    # INITIAL_TIMESTAMP = "hello"

def mongoUpload(data, mGClient):
    global INITIAL_TIMESTAMP
    db = mGClient['experiments']  # Replace 'your_database_name' with your actual database name
    # collection = db['your_collection_name']  # Replace 'your_collection_name' with your actual collection name
    # collection or Metadati or XsensSamples or final
    global LAST_XSENS_DATA
    LAST_XSENS_DATA = time.time()

     
    collection = db[f'Exp_{EXPERIMENTLABEL}_xsens']
    dataaaa = data["xsens_batch"]
    collection.insert_many(dataaaa)

    # db = client['xsensmsg']
    collection = db[f'Exp_{EXPERIMENTLABEL}_btsData']
    
    dataaaa = data["bts_batch"]
    if len(dataaaa) == 0:
        print("Empty bts batch")
    else:
        collection.insert_many(dataaaa)
    
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

def on_connect(client, userdata, flags, rc, properties):
    print("Connected with result code "+str(rc))
    client.subscribe("ExperimentInfo",  qos=1)

def on_message(client, userdata, msg):
    global Mongoclient
    global EXPERIMENTLABEL
    

    global MQTTADDRESS
    global MQTTPORT
    global VALORESCORSO
    global INITIAL_TIMESTAMP
    global XSENS_TIMEOUT
    if msg.topic == "ExperimentInfo":
        # Avvio dell'esperimento
        mess = json.loads(msg.payload)
        EXPERIMENTLABEL = mess["experimentLabel"]
        mongoMeta(mess, Mongoclient)

        print("Experiment started.")
        client.subscribe("$share/storage/SamplesData", qos=1)
        XSENS_TIMEOUT = 1
    elif msg.topic == "SamplesData":

        XSENS_TIMEOUT = 1
        
        try:
            data = msg.payload
            data = data.decode('utf-8')
            data = json.loads(data)
            mongoUpload(data, Mongoclient)
            
            
            print("Received a message from the broker: ")

            if VALORESCORSO == msg.payload:
                print("Stesso valore")
                raise KeyboardInterrupt
            else:   
                VALORESCORSO = msg.payload


            
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit()
        

if __name__ == "__main__":
    
    while True:
        print(f"Connecting to {MQTTADDRESS}:{MQTTPORT}")
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="storage")
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(MQTTADDRESS, MQTTPORT)
        VALORESCORSO = ""
        PREDICTIONS = []
        NUOVO_VALORE = None
        EXPERIMENTLABEL = None

        LAST_XSENS_DATA = None
        try:
            # You can add additional logic here before starting the loop if needed
            while True:
                if check_xsens_data_timeout():
                    break
                client.loop()

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit()

