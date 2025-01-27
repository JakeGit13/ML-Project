import pandas as pd
import numpy as np
from scipy import stats
import pickle

class SimpleAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01):
        # Initialize weights and learning rate
        self.learning_rate = learning_rate
        
        # Xavier/Glorot initialization
        self.encoder_weights = np.random.randn(input_dim, encoding_dim) * np.sqrt(2.0 / (input_dim + encoding_dim))
        self.encoder_bias = np.zeros(encoding_dim)
        self.decoder_weights = np.random.randn(encoding_dim, input_dim) * np.sqrt(2.0 / (input_dim + encoding_dim))
        self.decoder_bias = np.zeros(input_dim)
        
        print(f"\nAutoencoder initialized:")
        print(f"Input dimension: {input_dim}")
        print(f"Encoded dimension: {encoding_dim}")
        print(f"Learning rate: {learning_rate}")
    
    def sigmoid(self, x):
        """Compute sigmoid with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Compute derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
        sx = self.sigmoid(x)
        return sx * (1 - sx)
    
    def encode(self, X):
        """Forward pass through encoder."""
        self.encoder_input = X
        self.encoder_z = np.dot(X, self.encoder_weights) + self.encoder_bias
        encoded = self.sigmoid(self.encoder_z)
        return encoded
    
    def decode(self, encoded):
        """Forward pass through decoder."""
        self.decoder_input = encoded
        self.decoder_z = np.dot(encoded, self.decoder_weights) + self.decoder_bias
        decoded = self.sigmoid(self.decoder_z)
        return decoded
    
    def forward_pass(self, X):
        """Complete forward pass through the autoencoder."""
        self.encoded = self.encode(X)
        self.decoded = self.decode(self.encoded)
        return self.encoded, self.decoded
    
    def backward_pass(self, X):
        """
        Compute gradients using chain rule.
        Args:
            X: Original input data
        """
        m = X.shape[0]  # batch size
        
        # Compute gradients for decoder
        decoded_error = self.decoded - X  # dL/dy
        decoder_z_grad = decoded_error * self.sigmoid_derivative(self.decoder_z)  # dL/dz = dL/dy * dy/dz
        
        # Gradients for decoder parameters
        self.decoder_weights_grad = np.dot(self.decoder_input.T, decoder_z_grad) / m
        self.decoder_bias_grad = np.mean(decoder_z_grad, axis=0)
        
        # Compute gradients for encoder
        encoder_output_grad = np.dot(decoder_z_grad, self.decoder_weights.T)  # dL/dh
        encoder_z_grad = encoder_output_grad * self.sigmoid_derivative(self.encoder_z)  # dL/dz = dL/dh * dh/dz
        
        # Gradients for encoder parameters
        self.encoder_weights_grad = np.dot(self.encoder_input.T, encoder_z_grad) / m
        self.encoder_bias_grad = np.mean(encoder_z_grad, axis=0)
    
    def update_parameters(self):
        """Update weights and biases using computed gradients."""
        # Update encoder parameters
        self.encoder_weights -= self.learning_rate * self.encoder_weights_grad
        self.encoder_bias -= self.learning_rate * self.encoder_bias_grad
        
        # Update decoder parameters
        self.decoder_weights -= self.learning_rate * self.decoder_weights_grad
        self.decoder_bias -= self.learning_rate * self.decoder_bias_grad
    
    def compute_loss(self, X, reconstructed):
        """Compute mean squared error loss."""
        return np.mean((X - reconstructed) ** 2)
    
    def train(self, X, epochs=100, batch_size=32, verbose=True):
        """
        Train the autoencoder.
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
        """
        n_samples = X.shape[0]
        history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            # Mini-batch training
            batch_losses = []
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i + batch_size]
                
                # Forward pass
                _, reconstructed = self.forward_pass(batch)
                
                # Compute loss
                loss = self.compute_loss(batch, reconstructed)
                batch_losses.append(loss)
                
                # Backward pass
                self.backward_pass(batch)
                
                # Update parameters
                self.update_parameters()
            
            # Record epoch loss
            epoch_loss = np.mean(batch_losses)
            history.append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history

# Load and prepare data
print("Loading data...")
df = pd.read_csv('butterfly_data.csv')

# Select features
features = [
    'Average Elevation (m)',
    'Average annual precipitiation (mm)',
    'Mean temp in C',
    'Extent of Forest (1000 ha)',
    'area (sq km)',
    'latitude',
    'island'
]

# Clean and convert data
for feature in features:
    df[feature] = df[feature].astype(str).str.replace(',', '').astype(float)

# Create and normalize feature matrix
X = df[features].values
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Create and train autoencoder
input_dim = len(features)
encoding_dim = 2
autoencoder = SimpleAutoencoder(input_dim, encoding_dim, learning_rate=0.01)

# Train the model
print("\nStarting training...")
history = autoencoder.train(X_normalized, epochs=100, batch_size=8, verbose=True)

# Final evaluation
encoded_features, reconstructed_data = autoencoder.forward_pass(X_normalized)
final_mse = autoencoder.compute_loss(X_normalized, reconstructed_data)
print(f"\nFinal reconstruction MSE: {final_mse:.4f}")

# Create DataFrame with original features and encoded dimensions
analysis_df = pd.DataFrame(X_normalized, columns=features)
analysis_df['encoded_dim_1'] = encoded_features[:, 0]
analysis_df['encoded_dim_2'] = encoded_features[:, 1]
analysis_df['species_richness'] = df['Number of species']

# Print encoded features for analysis
print("\nFinal encoded features (first 5 samples):")
print(encoded_features[:5])

# Analyze correlations between encoded dimensions and species count
print("\nCorrelations between encoded features and species count:")
for col in ['encoded_dim_1', 'encoded_dim_2']:
    correlation = analysis_df[col].corr(analysis_df['species_richness'])
    print(f"{col}: {correlation:.4f}")

# Analyze correlations between encoded dimensions and original features
print("\nCorrelations with Encoded Dimension 1:")
for feature in features:
    correlation = stats.pearsonr(analysis_df[feature], analysis_df['encoded_dim_1'])
    print(f"{feature}: {correlation[0]:.4f} (p={correlation[1]:.4f})")

print("\nCorrelations with Encoded Dimension 2:")
for feature in features:
    correlation = stats.pearsonr(analysis_df[feature], analysis_df['encoded_dim_2'])
    print(f"{feature}: {correlation[0]:.4f} (p={correlation[1]:.4f})")

# Analyze feature importance through encoder weights
print("\nFeature importance based on encoder weights:")
for i, feature in enumerate(features):
    # Calculate absolute weight contribution for each feature
    importance_dim1 = abs(autoencoder.encoder_weights[i, 0])
    importance_dim2 = abs(autoencoder.encoder_weights[i, 1])
    print(f"\n{feature}:")
    print(f"  Contribution to dim 1: {importance_dim1:.4f}")
    print(f"  Contribution to dim 2: {importance_dim2:.4f}")

# Calculate reconstruction error per feature
reconstructed_df = pd.DataFrame(reconstructed_data, columns=features)
print("\nReconstruction error per feature:")
for feature in features:
    mse = np.mean((analysis_df[feature] - reconstructed_df[feature]) ** 2)
    print(f"{feature}: {mse:.4f}")



    # Add this at the end of your existing autoencoder script
import pickle

# Create a dictionary with all the data we need for visualization
visualization_data = {
    'history': history,
    'features': features,
    'encoded_features': encoded_features,
    'reconstructed_data': reconstructed_data,
    'analysis_df': analysis_df,
    'species_richness': df['Number of species'].values
}

# Save the data
print("\nSaving data for visualization...")
with open('autoencoder_results.pkl', 'wb') as f:
    pickle.dump(visualization_data, f)
print("Data saved successfully!")