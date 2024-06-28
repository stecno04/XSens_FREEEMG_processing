import tkinter as tk
import logging
 
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
 
# Creazione dell'handler per il file di log
file_handler = logging.FileHandler('.bts_analisy_log/effort_logs.log')
file_handler.setLevel(logging.DEBUG)
 
# Creazione di un formatter personalizzato con formato di timestamp ISO 8601
formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
 
# Aggiunta del formatter all'handler
file_handler.setFormatter(formatter)
 
# Aggiunta dell'handler al logger
logger.addHandler(file_handler)
 
 
def on_excesive_effort_click():
    global logger
    logger.warning("Stanco")
    print('\a')
 
root = tk.Tk()
root.title("Tkinter Example")
 
 
label = tk.Label(root, text="Fai click nel bottone quando tu sia stanco!. Tutti i click fatti saranno salvatti nel file di logs")
label.pack()
 
button = tk.Button(root, text="Sono stanchissimoooo", command=on_excesive_effort_click)
button.pack()
 
root.mainloop()