from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from ds_manager import scrivi_dataset_binario
import numpy as np

# Conversione dataset Collision in formato binario
#df = pd.read_excel('TabularData\collision\collision.xlsx')
# Converti in array numpy
#dataset = df.values



# Conversione dataset DryBean in formato binario
drybean = fetch_ucirepo(id=602)
# Ottieni i dati come array numpy
features = drybean.data.features.values
target = drybean.data.targets.values
dataset = np.hstack((features, target))

# Converti le stringhe del target in numeri
label_encoder = LabelEncoder()
target_numeric = label_encoder.fit_transform(target.ravel())
# target.ravel(): converte il target da una matrice 2D (n,1) in un array 1D (n,) appiattendo i dati

# Combina features e target, con il target come ultima colonna
dataset = np.hstack((features, target_numeric.reshape(-1, 1)))
# target_numeric.reshape(-1, 1): converte l'array 1D in una matrice colonna (n,1)

# Stampa il mapping
print("\nMapping delle etichette:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")


# Salva in formato binario
scrivi_dataset_binario('drybean.ds3', dataset)

print(f"Dataset convertito e salvato in formato binario.")
print(f"Dimensioni originali: {dataset.shape}") 
