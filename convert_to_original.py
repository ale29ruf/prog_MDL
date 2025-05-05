import numpy as np
import pandas as pd

from ds_manager import leggi_dataset_binario

dataset = leggi_dataset_binario('condensed.ds3')

features = dataset[:, :-1]  
target = dataset[:, -1] 

df = pd.DataFrame(features)
df['Class'] = target

df.to_excel('condensed_original.xlsx', index=False)

print("Dataset convertito e salvato in formato Excel.")
print(f"Dimensioni features: {features.shape}")
print(f"Numero di classi: {len(np.unique(target))}")
print("\nClassi presenti:")
for label in np.unique(target):
    count = np.sum(target == label)
    print(f"Classe {label}: {count} esempi") 