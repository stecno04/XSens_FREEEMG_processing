import numpy as np
import re
import json 

def Rotation(q0, q1, q2, q3, x, y, z):
    """
    Returns the rotation matrix
        Inputs:
            q0: quaternion value
            q1: quaternion value
            q2: quaternion value
            q3: quaternion value
            x: x value
            y: y value
            z: z value
        Outputs:
            T: the rotation matrix
    """
    R_pelv = np.array([[1-(2*(q2**2+q3**2)), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
                        [2 * (q1*q2 + q0*q3), 1-(2*(q1**2+q3**2)), 2 * (q2*q3 - q0*q1)],
                        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1-(2*(q1**2+q2**2))]])

    quart_colonna = np.dot(R_pelv.T, np.array([x, y, z]))

    T = np.eye(4)
    T[:3, 3] = quart_colonna
    T[:3, :3] = R_pelv
    
    return T
def quaternioni(datas):
    """
    Returns the rotation matrix and the updated dictionary
        Inputs:
            datas: the dictionary with the data
        Outputs:
            QuatPelvisR: the rotation matrix
            datas: the updated dictionary
    """

    q0 = datas['Segment Orientation - Quat_Pelvis q0']
    q1 = datas['Segment Orientation - Quat_Pelvis q1']
    q2 = datas['Segment Orientation - Quat_Pelvis q2']
    q3 = datas['Segment Orientation - Quat_Pelvis q3']
    x = datas['Segment Position_Pelvis x']
    y = datas['Segment Position_Pelvis y']
    z = datas['Segment Position_Pelvis z']
        
    # Calculate the rotation matrix
    rotation_matrix = Rotation(q0, q1, q2, q3, x, y, z)
    datas.pop('Segment Orientation - Quat_Pelvis q0')
    datas.pop('Segment Orientation - Quat_Pelvis q1')
    datas.pop('Segment Orientation - Quat_Pelvis q2')
    datas.pop('Segment Orientation - Quat_Pelvis q3')
    datas.pop('Segment Position_Pelvis x')
    datas.pop('Segment Position_Pelvis y')
    datas.pop('Segment Position_Pelvis z')

    # Assign the rotation matrix to the DataFrame
    QuatPelvisR = rotation_matrix
    return QuatPelvisR, datas
def extract_string(text):
    """
    Returns the extracted string
        Inputs:
            text: the text to extract the string from
        Outputs:
            match.group(1): the extracted string
    """
    match = re.search(r'_(.*?)(?=\w$)', text)
    if match:
        return match.group(1) 
def T(R, x, y, z):
    """
    Returns the transformation matrix
        Inputs:
            R: the rotation matrix
            x: x value
            y: y value
            z: z value
        Outputs:
            array_string: the transformation matrix
    """
    matrice = np.array(R)

    # Calcola la trasposta della matrice
    matrice_trasposta = np.transpose(matrice)


    # Definisci il vettore
    vettore = np.array([x, y, z, 1])

    # Moltiplica la trasposta della matrice per il vettore
    risultato = np.dot(matrice_trasposta, vettore)

    # Convert the array to a string representation
    array_string = str(risultato.tolist())


    return array_string
def ternions(datas, RMat):
    """
    Returns the updated dictionary, the final dictionary 
        Inputs:
            datas: the dictionary with the data
            RMat: the rotation matrix
        Outputs:
            datas: the updated dictionary
            final: the final dictionary
    """
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

    listsens = []
    final = {}
    for col in datas.keys():  # Accedi alle chiavi del dizionario
        if col in listacolon:
            match = re.match(r'(.+) ', col)
            if match:
                substring = match.group(1)
                if substring not in listsens:
                    listsens.append(substring)
                    x = datas[substring + ' x']
                    y = datas[substring + ' y']
                    z = datas[substring + ' z']
                    # Aggiorna il dizionario 'elena' con i nuovi valori
                    elemento = T(RMat, x, y, z)
                    final[f'Segment Position_{extract_string(col)}'] = elemento

    

    daRimuovere =['Segment Orientation - Quat_Frame', 'Segment Position_L5 x', 'Segment Position_L5 y', 'Segment Position_L5 z', 'Segment Position_L3 x', 'Segment Position_L3 y', 'Segment Position_L3 z', 'Segment Position_T12 x', 'Segment Position_T12 y', 'Segment Position_T12 z', 'Segment Position_T8 x', 'Segment Position_T8 y', 'Segment Position_T8 z', 'Segment Position_Neck x', 'Segment Position_Neck y', 'Segment Position_Neck z', 'Segment Position_Head x', 'Segment Position_Head y', 'Segment Position_Head z', 'Segment Position_Right Shoulder x', 'Segment Position_Right Shoulder y', 'Segment Position_Right Shoulder z', 'Segment Position_Right Upper Arm x', 'Segment Position_Right Upper Arm y', 'Segment Position_Right Upper Arm z', 'Segment Position_Right Forearm x', 'Segment Position_Right Forearm y', 'Segment Position_Right Forearm z', 'Segment Position_Left Shoulder x', 'Segment Position_Left Shoulder y', 'Segment Position_Left Shoulder z', 'Segment Position_Left Upper Arm x', 'Segment Position_Left Upper Arm y', 'Segment Position_Left Upper Arm z', 'Segment Position_Left Forearm x', 'Segment Position_Left Forearm y', 'Segment Position_Left Forearm z']
    
    for col in daRimuovere:
        try:

            popped = datas.pop(col)
        except:
            pass
        
    return datas, final
def xyzSeparazione(datas):
    """
    Returns the dictionary with the separated values
        Inputs:
            datas: the dictionary with the data
            Outputs:
                output_dict: the dictionary with the separated values
    """
    output_dict = {}

    # Regex per estrarre i numeri da una stringa
    pattern = re.compile(r'-?\d+\.\d+')

    for key, value in datas.items():
        # Rimuovi le parentesi quadre e splitta la stringa in numeri
        numbers = re.findall(pattern, value)
        
        # Crea le chiavi per x, y, z
        key_x = key.strip() + '_x'
        key_y = key.strip() + '_y'
        key_z = key.strip() + '_z'
        
        # Aggiungi i valori al nuovo dizionario
        output_dict[key_x] = float(numbers[0])
        output_dict[key_y] = float(numbers[1])
        output_dict[key_z] = float(numbers[2])

    return output_dict

def predizione(NumpyDf, model, pca):
    """
    Returns the predicted clusters based on the input data
        Inputs:
            NumpyDf: the input data
            model: the clustering model
            pca: the PCA model
        Outputs:
            predicted_clusters: the predicted clusters
    """
    # Misura il tempo di esecuzione della trasformazione PCA
    X_pca_new = pca.transform(NumpyDf)[:, :2]

    # Misura il tempo di esecuzione della previsione dei cluster
    predicted_clusters = model.predict(X_pca_new)
    # Aggiungi una colonna al dataframe contenente i cluster assegnati
    return predicted_clusters

