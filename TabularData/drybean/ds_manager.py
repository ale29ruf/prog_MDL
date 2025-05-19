import struct
import numpy as np



def leggi_dataset_binario(percorso_file):

    with open(percorso_file, 'rb') as f:

        # Legge i primi 8 byte: due int32
        intestazione = f.read(8)
        colonne, righe = struct.unpack('ii', intestazione)
        print(f"Numero di righe: {righe}, colonne: {colonne}")

        # Calcola quanti float32 devono essere letti
        num_elementi = righe * colonne

        # Legge i dati restanti come float32
        dati_binari = f.read(num_elementi * 4)  # 4 byte per ogni float32
        dati = np.frombuffer(dati_binari, dtype=np.float32)

        # Reshape in matrice
        dataset = dati.reshape((righe, colonne))

        return dataset


def scrivi_dataset_binario(percorso_file, dataset):
    righe, colonne = dataset.shape

    # Assicura che i dati siano in formato float32
    dataset = dataset.astype(np.float32)

    with open(percorso_file, 'wb') as f:
        # Scrive intestazione: numero di righe e colonne come int32
        intestazione = struct.pack('ii', colonne, righe)
        f.write(intestazione)

        # Scrive i dati come float32
        f.write(dataset.tobytes())

    return

