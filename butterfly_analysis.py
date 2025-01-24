import pandas as pd
import numpy as np

# 1. Load and prepare data
print("Loading data...")
df = pd.read_csv('butterfly_data.csv')
print("\nOriginal data shape:", df.shape)

# 2. Select features with exact column names
features = [
    'Average Elevation (m)',
    'Average annual precipitiation (mm)',
    'Mean temp in C',
    'Extent of Forest (1000 ha)',
    'area (sq km)',
    'latitude',
    'island'
]

# 3. Clean and convert data to numeric
for feature in features:
    # Remove commas and convert to numeric
    df[feature] = df[feature].astype(str).str.replace(',', '').astype(float)

# 4. Create feature matrix X
X = df[features].values
print("\nFeature matrix shape:", X.shape)

# 5. Print sample of data to verify it's all numerical
print("\nFirst few rows of feature matrix:")
print(X[:5])

# 6. Normalize the features to [0,1] range
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("\nVerifying normalization - range for each feature:")
for i, feature in enumerate(features):
    print(f"{feature}: [{X_normalized[:,i].min():.2f}, {X_normalized[:,i].max():.2f}]")

# 7. Basic autoencoder class
class SimpleAutoencoder:
    def __init__(self, input_dim, encoding_dim):
        # Initialize random weights for encoder
        self.encoder_weights = np.random.randn(input_dim, encoding_dim) * 0.01
        self.encoder_bias = np.zeros(encoding_dim)
        
        # Initialize random weights for decoder
        self.decoder_weights = np.random.randn(encoding_dim, input_dim) * 0.01
        self.decoder_bias = np.zeros(input_dim)
        
        print(f"\nAutoencoder initialized:")
        print(f"Input dimension: {input_dim}")
        print(f"Encoded dimension: {encoding_dim}")

# 8. Create an instance
input_dim = len(features)  # number of features
encoding_dim = 2  # compress to 2 dimensions
autoencoder = SimpleAutoencoder(input_dim, encoding_dim)