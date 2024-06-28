# Xsens Ricezione, analisi e invio dati

## Introduzione
Il progetto è stato sviluppato per la tesi di laurea di un membro del team. Lo scopo è quello di ricevere i dati da una serie di sensori inerziali Xsens AWINDA e dai sensori elletromiografici FREEEMG1000, analizzarli e inviarli ad un server remoto, nel nostro caso MongoDB. Il progetto è stato sviluppato principalmente in Python ma può essere eseguito anche in Docker. Il progetto è stato sviluppato in modo da essere facilmente estendibile e modificabile. Nella repository possiamo trovare programmi di training, testing e analisi dati.

## Prerequisiti

- python 3.11 (si racccomanda l'uso di un virtual environment)
- Docker 4.29.0 o versioni successive (opzionale). 
- Acesso ad un server MongoDB
- Acesso ad un server MQTT
- Acesso ad un server MLFLOW (opzionale)

Per installare le librerie python necessarie al progetto è sufficiente eseguire il seguente comando:
```bash
pip install -r requirements.txt
```
## Tree
```
.
├── README.MD
├── requirements.txt
├── dockerfile
└── notebooks: Programmi sviluppati per per testare il training di diversi modelli sui sensori XSens e non più utilizzati
    |── danishNotebook.ipynb
    ├── envelopeBTS.ipynb
    ├── integrale.py
    ├── niceGraphsNN.ipynb
    ├── WhichDB.ipynb
    └── trainingMisureCorpo.py
├── funzionante_training: Programmi utilizzati per il training del progetto il progetto 
    ├── trainingSenzaMisure.py
    └── trainingMisureCorpo.py
├── liveProcessing: Programmi sviluppati per il progetto e separati in storage e processing
    ├── processing.py programmi per il processing dei dati con shared mqtt
    └── storage.py
└── analysis_functions: Libreria sviluppata per analisi e RMS sui dati EMG
    ├── __init__.py
    ├── XSensAnalysis.py
    └── ebtsAnalysis.py
```

## Utilizzo 
### Training
Al momento esistono due programmi principali per il training del modello. Il primo programma è `trainingSenzaMisure.py` che permette di allenare il modello con i dati ricevuti dai sensori XSens e salvati su una cartella chiamata `data` in formato Excel. Il programma prendendo come input direttamente i dati estratti dall'applicazione Xsens riesce prima a trainare un modello PCA e tramite questo output traina una neural Network. Entrambi i modelli possono essere salvati in locale e/o su un server MLFLOW. 
Per eseguire il training del modello è sufficiente eseguire il seguente comando:
```bash
python funzionante_training/trainingSenzaMisure.py
```
è possibile aggiungere i seguenti parametri:
- `--mlflow` per salvare il modello su un server MLFLOW (opzionale)
- `--no-local` per non salvare il modello in locale (opzionale)
- `--mlfow_uri` per specificare l'indirizzo del server MLFLOW (default:http://10.250.4.35:8000)
- `--mlfow_experiment` per specificare l'esperimento su cui salvare il modello (default: XSense Training)



Il secondo programma è `TrainingConMisure.py` che permette di allenare il modello con i dati ricevuti dai sensori XSens e salvati su un server MongoDB. Per eseguire il training del modello è sufficiente eseguire il seguente comando, bisogna però avere e settare manualmente i nomi dei vari esperimenti ed essendo training supervisionato bisogna avere anche i valori target per ogni esperimento inserendoli manualmente a riga 167:

```bash
python TrainingConMisure.py
```
### Live Processing
Al momento il processing è diviso in due tipi di processi:

1) Il primo file `storage.py` permette di ricevere i dati da MQTT e salvarli direttamente su un server MongoDB senza fare alcun processing. Per eseguire quindi lo storage dei dati è sufficiente eseguire il seguente comando:
```bash
python liveProcessing/storage.py
```
è possibile aggiungere i seguenti parametri:
- `--mongodb_uri` per specificare l'indirizzo del server MongoDB (default: mongodb://localhost:27017)
- `--mqtt_port` per specificare la porta del server MQTT (default: 1883)
- `--mqtt_address` per specificare l'indirizzo del server MQTT (default: '10.250.4.35')

2) Il secondo file `processing.py` permette di ricevere i dati da MQTT, fare il processing dei dati e salvarli su un server MongoDB . Per eseguire quindi il processing dei dati è sufficiente eseguire il seguente comando:
```bash
python liveProcessing/processing.py
```
è possibile aggiungere i seguenti parametri:
- `--mongodb_uri` per specificare l'indirizzo del server MongoDB (default: mongodb://10.250.4.35:27017)
- `--mqtt_port` per specificare la porta del server MQTT (default: 1883)
- `--mqtt_address` per specificare l'indirizzo del server MQTT (default: 10.250.4.35)
- `--mlflow_uri` per specificare l'indirizzo del server MLFLOW (default: http://10.250.4.35:8000)
- `--mlflow_experiment` per specificare l'esperimento su cui salvare il modello (default: XSense Training)
- `--local` per utilizzare il modello salvato in locale (default: False)
- `--using_specific_model` per utilizzare un modello specifico di MLFLOW (default: False)

Questi programmi fanno uso della feature **shared subscribe** presente nel protocollo MQTT v5. Questa feature permette di avere più client che si connettono allo stesso topic e ricevono un messaggio a testa, molto utili qualora il processo fosse troppo lungo per una questione di `Load Balancing`. Applicato a questo progetto questa funzione ci permette di eseguire diversi script contemporaneamente e distribuire il carico di lavoro.

### Docker
Il progetto è stato sviluppato in modo da essere eseguito anche in Docker. Per costruire l'immagine docker è sufficiente eseguire il seguente comando:
```bash
docker build -t ghcr.io/facilitiesmade/xsens_processing:latest -t ghcr.io/facilitiesmade/xsens_processing:1.0 .
```
per eseguire un container che faccia il lavoro di processing bisogna eseguire il seguente comando:

```bash
docker run -d --name xsens_processing ghcr.io/facilitiesmade/xsens_processing:latest
```

per eseguire un container che faccia il lavoro di Storage bisogna eseguire il seguente comando:

```bash
docker run -d --name xsens_processing ghcr.io/facilitiesmade/xsens_processing:latest python liveProcessing/storage.py
```

Le immagine docker create posso essere messe nel docker registry di github e scaricate in qualsiasi macchina che abbia docker installato.
Per configurare l'accesso al container registry è necessario seguire il processo descritto [qui](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-docker-registry) usando come token quello salvato nella repository di Truppa in credenziali Aree.


## Conclusioni
Il progetto è ancora in sviluppo.
