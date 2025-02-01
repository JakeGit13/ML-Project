import pandas as pd
import numpy as np
from scipy import stats
import pickle

class SimpleAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01):
        """Initialize autoencoder with Xavier/Glorot initialization"""
        self.learning_rate = learning_rate
        scale = np.sqrt(2.0 / (input_dim + encoding_dim))
        
        # Simplified weight initialization
        self.encoder_weights = np.random.randn(input_dim, encoding_dim) * scale
        self.decoder_weights = np.random.randn(encoding_dim, input_dim) * scale
        self.encoder_bias = np.zeros(encoding_dim)
        self.decoder_bias = np.zeros(input_dim)
    
    def sigmoid(self, x):
        """Compute sigmoid with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Compute derivative of sigmoid"""
        sx = self.sigmoid(x)
        return sx * (1 - sx)
    
    def forward_pass(self, X):
        """Combined forward pass through the autoencoder"""
        # Encode
        self.encoder_input = X
        self.encoder_z = np.dot(X, self.encoder_weights) + self.encoder_bias
        self.encoded = self.sigmoid(self.encoder_z)
        
        # Decode
        self.decoder_z = np.dot(self.encoded, self.decoder_weights) + self.decoder_bias
        self.decoded = self.sigmoid(self.decoder_z)
        
        return self.encoded, self.decoded
    
    def backward_pass(self, X):
        """Compute gradients using chain rule"""
        m = X.shape[0]
        
        # Compute all gradients
        decoded_error = self.decoded - X
        decoder_z_grad = decoded_error * self.sigmoid_derivative(self.decoder_z)
        
        self.decoder_weights_grad = np.dot(self.encoded.T, decoder_z_grad) / m
        self.decoder_bias_grad = np.mean(decoder_z_grad, axis=0)
        
        encoder_output_grad = np.dot(decoder_z_grad, self.decoder_weights.T)
        encoder_z_grad = encoder_output_grad * self.sigmoid_derivative(self.encoder_z)
        
        self.encoder_weights_grad = np.dot(self.encoder_input.T, encoder_z_grad) / m
        self.encoder_bias_grad = np.mean(encoder_z_grad, axis=0)
    
    def update_parameters(self):
        """Update network parameters"""
        self.encoder_weights -= self.learning_rate * self.encoder_weights_grad
        self.encoder_bias -= self.learning_rate * self.encoder_bias_grad
        self.decoder_weights -= self.learning_rate * self.decoder_weights_grad
        self.decoder_bias -= self.learning_rate * self.decoder_bias_grad
    
    def compute_loss(self, X, reconstructed):
        """Compute MSE loss"""
        return np.mean((X - reconstructed) ** 2)
    
    def split_train_val(self, X, val_fraction=0.2):
        """Split data into training and validation sets"""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        val_size = int(val_fraction * n_samples)
        return X[indices[val_size:]], X[indices[:val_size]]
    
    def train(self, X, epochs=100, batch_size=32, verbose=True, val_fraction=0.2):
        """Train the autoencoder with integrated early stopping"""
        if verbose:
            print(f"\nTraining on {len(X)} samples ({int((1-val_fraction)*len(X))} train, {int(val_fraction*len(X))} validation)")
        
        X_train, X_val = self.split_train_val(X, val_fraction)
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience, patience_counter = 5, 0
        min_delta = 0.001
        
        for epoch in range(epochs):
            # Training step
            indices = np.random.permutation(n_samples)
            batch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch = X_train[indices[i:i + batch_size]]
                _, reconstructed = self.forward_pass(batch)
                batch_losses.append(self.compute_loss(batch, reconstructed))
                self.backward_pass(batch)
                self.update_parameters()
            
            # Compute losses
            train_loss = np.mean(batch_losses)
            _, val_reconstructed = self.forward_pass(X_val)
            val_loss = self.compute_loss(X_val, val_reconstructed)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Progress tracking
            if verbose and ((epoch + 1) % 10 == 0 or val_loss < best_val_loss - min_delta):
                print(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                    break
        
        if verbose:
            print(f"\nTraining completed after {epoch + 1} epochs")
            print(f"Final losses - train: {train_loss:.4f}, val: {val_loss:.4f}")
        
        return history

def analyze_results(autoencoder, X_normalized, df, features):
    """Analyze autoencoder results"""
    encoded_features, reconstructed_data = autoencoder.forward_pass(X_normalized)
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame(X_normalized, columns=features)
    analysis_df['encoded_dim_1'] = encoded_features[:, 0]
    analysis_df['encoded_dim_2'] = encoded_features[:, 1]
    analysis_df['species_richness'] = df['Number of species']
    
    # Analyze correlations
    print("\nCorrelations with species count:")
    for dim in ['encoded_dim_1', 'encoded_dim_2']:
        corr = analysis_df[dim].corr(analysis_df['species_richness'])
        print(f"{dim}: {corr:.4f}")
    
    # Feature correlations and importance
    for feature in features:
        dim1_corr = stats.pearsonr(analysis_df[feature], analysis_df['encoded_dim_1'])
        dim2_corr = stats.pearsonr(analysis_df[feature], analysis_df['encoded_dim_2'])
        importance_dim1 = abs(autoencoder.encoder_weights[features.index(feature), 0])
        importance_dim2 = abs(autoencoder.encoder_weights[features.index(feature), 1])
        
        print(f"\n{feature}:")
        print(f"  Correlation (dim1): {dim1_corr[0]:.4f} (p={dim1_corr[1]:.4f})")
        print(f"  Correlation (dim2): {dim2_corr[0]:.4f} (p={dim2_corr[1]:.4f})")
        print(f"  Importance: dim1={importance_dim1:.4f}, dim2={importance_dim2:.4f}")
    
    return encoded_features, reconstructed_data, analysis_df

def main():
    # Data preparation
    print("Loading and preparing data...")
    df = pd.read_csv('butterfly_data.csv')
    
    features = [
        'Average Elevation (m)',
        'Average annual precipitiation (mm)',
        'Mean temp in C',
        'Extent of Forest (1000 ha)',
        'area (sq km)',
        'latitude',
        'island',
        'Population density (sq km)'
    ]
    
    print("\nCleaning data...")
    for feature in features:
        df[feature] = df[feature].astype(str).str.replace(',', '').astype(float)
    X = df[features].values

    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Train autoencoder
    autoencoder = SimpleAutoencoder(len(features), 2, learning_rate=0.01)
    history = autoencoder.train(X_normalized, epochs=100, batch_size=8, verbose=True)
    
    # Analyze and save results
    encoded_features, reconstructed_data, analysis_df = analyze_results(autoencoder, X_normalized, df, features)
    
    visualization_data = {
        'history': history,
        'features': features,
        'encoded_features': encoded_features,
        'reconstructed_data': reconstructed_data,
        'analysis_df': analysis_df,
        'species_richness': df['Number of species'].values
    }
    
    print("\nSaving results...")
    with open('autoencoder_results.pkl', 'wb') as f:
        pickle.dump(visualization_data, f)
    print("Analysis complete!")

if __name__ == "__main__":
    main()