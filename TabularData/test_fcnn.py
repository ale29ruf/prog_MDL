import pandas as pd
import numpy as np

from fcnn import FCNN

# Load the dataset
df = pd.read_excel('TabularData/collision/collision.xlsx') # /kaggle/input/collisionfcnn/collision.xlsx

# Separate features and target
X = df.drop('collision', axis=1).values
y = df['collision'].values

# Define alpha values to test
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Test FCNN with different alpha values
for alpha in alpha_values:
    print(f"\nTesting FCNN with alpha = {alpha}")
    fcnn = FCNN()
    subset, subset_labels, reduced_ratio = fcnn.fit(X, y, alpha=alpha)
    
    print(f"Subset shape: {subset.shape}")
    print(f"Number of unique labels in subset: {len(np.unique(subset_labels))}")
    print(f"Subset labels distribution:")
    unique_labels, counts = np.unique(subset_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} instances") 