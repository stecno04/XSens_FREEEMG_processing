import numpy as np
import pandas as pd
import datetime

import bson

import json
import pymongo
from pymongo import MongoClient

import pymongo.collection
from scipy import signal
from collections import defaultdict
from typing import Tuple

Sampling_frequency = 1000
band_lo = 20
band_hi = 450
Niquist_frequency = Sampling_frequency/2
nor_band_lo = band_lo/Niquist_frequency
nor_band_hi = band_hi/Niquist_frequency
sos_band = signal.iirfilter(4, [ nor_band_lo, nor_band_hi],btype='band', ftype='butter', output = "sos")
del_freq = 50
nor_del_freq = del_freq/Niquist_frequency
Quality = 30
b_notch, a_notch = signal.iirnotch(nor_del_freq, Quality, Sampling_frequency)

def obtaining_mqtt(bts_batch: list, dataMongo: defaultdict) -> defaultdict:
    """
    Returns the data taking a maximum of 3000 samples for each sensor
        Inputs:
            bts_batch: list of dictionaries with the new data of the sensors
            dataMongo: database imported from mongoDB with the old data of the sensors
        Outputs:
            dati: database with the updated last 3000 samples for each sensor
    """
    print(dataMongo.keys())
    for key, value in dataMongo.items():
        if len(value)>=3000:
            dataMongo[key] = value[-2800:]

    for x in bts_batch:
        try:
            samples : list = x["samples"]
            for sample in samples:
                dataMongo[str(sample["sensorLabel"])].append((x["index"], sample["status"], sample["value"]))
        except:
            print(x)
    
    return dataMongo

def obtain_mongo(timestamp_exp: str,  collection: pymongo.collection.Collection) -> Tuple[dict, defaultdict]:
    """
    Returns the data from the database mongoDB
        Inputs:
            timestamp_exp: initial timestamp of the experiment from where we will find the relative object id
            collection: collection of the database mongoDB
        Outputs:
            data: data of the sensors old
    """
    dt = datetime.datetime.fromisoformat(timestamp_exp)
    obj_id = bson.ObjectId.from_datetime(dt)
    cursor = collection.find_one({"_id": obj_id})
    data = None
    if cursor is None:
        collection.insert_one({"_id": obj_id, "data": json.dumps({}), "integral": json.dumps({})})
        data = defaultdict(list)
        integralValue = defaultdict(float)
    else:
        data = defaultdict(list, json.loads(cursor["data"]))
        integralValue = defaultdict(float, json.loads(cursor["integral"]))
    # print(data.keys())
    return data, integralValue

def filtering(data: defaultdict) -> dict:
    """
    Returns the data filtered with bandpass and notch filters
        Inputs:
            data: database with the data of the sensors
        Outputs:
            filtered_data: database with the data of the sensors filtered
    """
    filtered_data = dict()
    for key, value in data.items():
        signal_array = np.array(value)
        signal_array = signal_array[signal_array[:,1] == 1][-2000:,:]

        signal_array_band = signal.sosfiltfilt(sos_band , signal_array[:,2])
        filtered = signal.lfilter(b_notch, a_notch, signal_array_band)
        signal_array[:,2] = filtered
        filtered_data[key] = signal_array
    print(data.keys())
    return filtered_data


def rms(data: defaultdict) -> dict:
    """
    Returns the RMS of the data
        Inputs:
            data: database with the data of the sensors
        Outputs:
            rms_data: database with the RMS of the data
    """
    rms_data = dict()
    for key, value in data.items():
        signal_array = np.array(value)
        rms = np.sqrt(np.mean(signal_array[:,2]**2))
        rms_data[key] = rms
    return rms_data

def meanMedianFreq(data: dict) -> dict:
    """
    Returns the mean frequency, mean power frequency and median frequency of the data
        Inputs:
            data: database with the data of the sensors
        Outputs:
            meanMedianFreq_data: database with the mean frequency, mean power frequency and median frequency of the data
    """
    meanMedianFreq_data = dict()
    for key, value in data.items():
        if value.shape[0] < 2000:
            print("data too short")
            continue

        print(value.shape)
        seg_i = value[:,2]
        frequency_domain = np.fft.fft(seg_i)
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(frequency_domain)
        # Identify frequency bins
        N = len(seg_i)
        sampling_rate = 1/1000
        frequency_bins = np.fft.fftfreq(N, d=sampling_rate)
        positive_frequencies = frequency_bins[:N//2]  # consider only positive frequencies
        mean_frequency = np.sum(magnitude_spectrum[:N//2] * positive_frequencies) / np.sum(magnitude_spectrum[:N//2])
        mean_power_frequency = np.sqrt(np.sum(magnitude_spectrum[:N//2] * positive_frequencies**2) / np.sum(magnitude_spectrum[:N//2]))
        # Calculate median frequency
        cumulative_power = np.cumsum(magnitude_spectrum[:N//2])
        half_power = np.sum(magnitude_spectrum[:N//2]) / 2
        median_index = np.argmax(cumulative_power > half_power)
        median_frequency = positive_frequencies[median_index]
        #   print(mean_power_frequency, mean_frequency, median_frequency)
        meanMedianFreq_data[key] =(mean_frequency, mean_power_frequency, median_frequency)
    return meanMedianFreq_data
    
def save_mongo(data: defaultdict, timestamp_exp: str, collection: pymongo.collection.Collection, ingregrale:defaultdict) -> None:
    """
    Saves the data in the database mongoDB
        Inputs:
            data: database with the data of the sensors
            timestamp_exp: timestamp of the experiment
            collection: collection of the database mongoDB
        Outputs:
            None
    """
    dt = datetime.datetime.fromisoformat(timestamp_exp)
    obj_id = bson.ObjectId.from_datetime(dt)
    collection.replace_one({"_id": obj_id},{"data": json.dumps(data), "integral": json.dumps(ingregrale)}, upsert=True)  

def integralBTS(data: defaultdict, integralValue: defaultdict, lastIndex: int) -> defaultdict:
    

    for key, value in data.items():
        signal_array = value[-200:, [0,2]]
        signal_array = signal_array[signal_array[:,0] >= lastIndex]
        if signal_array.size > 0:
            # prendiamo i valori di signal array e li moltiplichiamo per 0.001
            integrals = np.abs(signal_array[:,1]) * 0.001
            # calcoliamo la somma pessata dei valori in integrals
            # eg integrals = [1,2,3,4,5] exponents = [0.99996^4, 0.99996^3, 0.99996^2, 0.99996^1, 0.99996^0]
            exponents = np.power(np.ones([signal_array.shape[0]])*0.99996, 
                                 np.arange(signal_array.shape[0])[::-1])
            
            integral_sum = np.sum(integrals*exponents)

            integralValue[key] = integral_sum + integralValue[key] * 0.99996**len(signal_array)

            # for i in range(signal_array.shape[0]):
            #     integral_now = abs(signal_array[i,1])*0.001 + integralValue[key]*0.99996
            #     integralValue[key] = integral_now
        # data_batch = sens_1[i:i+200]
        # data_filt = [abs(k[2]) for k in data_batch if k[1]==1]
        # integrals.append(integrate.simpson(data_filt, dx=0.001))

    return integralValue