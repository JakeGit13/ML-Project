import pandas as pd
import numpy as np
from scipy import stats
import pickle

# Add this after your imports at the top of the file
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



class SimpleAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01):
        """
        Initialize the autoencoder with specified dimensions.
        
        Args:
            input_dim: Number of input features (8 in our case)
            encoding_dim: Size of the encoded representation (2 in our case)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights using Xavier/Glorot initialization
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
        """Compute derivative of sigmoid."""
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
        """Compute gradients using chain rule."""
        m = X.shape[0]  # batch size
        
        # Compute gradients for decoder
        decoded_error = self.decoded - X
        decoder_z_grad = decoded_error * self.sigmoid_derivative(self.decoder_z)
        
        # Gradients for decoder parameters
        self.decoder_weights_grad = np.dot(self.decoder_input.T, decoder_z_grad) / m
        self.decoder_bias_grad = np.mean(decoder_z_grad, axis=0)
        
        # Compute gradients for encoder
        encoder_output_grad = np.dot(decoder_z_grad, self.decoder_weights.T)
        encoder_z_grad = encoder_output_grad * self.sigmoid_derivative(self.encoder_z)
        
        # Gradients for encoder parameters
        self.encoder_weights_grad = np.dot(self.encoder_input.T, encoder_z_grad) / m
        self.encoder_bias_grad = np.mean(encoder_z_grad, axis=0)
    
    def update_parameters(self):
        """Update weights and biases using computed gradients."""
        self.encoder_weights -= self.learning_rate * self.encoder_weights_grad
        self.encoder_bias -= self.learning_rate * self.encoder_bias_grad
        self.decoder_weights -= self.learning_rate * self.decoder_weights_grad
        self.decoder_bias -= self.learning_rate * self.decoder_bias_grad
    
    def compute_loss(self, X, reconstructed):
        """Compute mean squared error loss."""
        return np.mean((X - reconstructed) ** 2)
    
    # Add this method before the train method in your SimpleAutoencoder class
    def split_train_val(self, X, val_fraction=0.2):
        """Split data into training and validation sets."""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        val_size = int(val_fraction * n_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return X[train_indices], X[val_indices]
        
    # Modify your train method to include these print statements
    # This should replace your existing train method, keeping all the early stopping logic we just added

    # Modify your train method to include these print statements
    # This should replace your existing train method, keeping all the early stopping logic we just added


    def train(self, X, epochs=100, batch_size=32, verbose=True, val_fraction=0.2):
        """
        Train the autoencoder with early stopping and detailed monitoring.
        """
        print("\n=== Starting Training Process ===")
        print(f"Training set size: {int((1-val_fraction)*len(X))} samples")
        print(f"Validation set size: {int(val_fraction*len(X))} samples")
        print("================================")
        
        # Split data into training and validation sets
        X_train, X_val = self.split_train_val(X, val_fraction)
        n_samples = X_train.shape[0]
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Shuffle and train on batches
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            batch_losses = []  # Initialize batch losses list
            
            for i in range(0, n_samples, batch_size):
                batch = X_train_shuffled[i:i + batch_size]
                _, reconstructed = self.forward_pass(batch)
                loss = self.compute_loss(batch, reconstructed)
                batch_losses.append(loss)
                self.backward_pass(batch)
                self.update_parameters()
            
            # Compute training and validation losses
            train_loss = np.mean(batch_losses)
            _, val_reconstructed = self.forward_pass(X_val)
            val_loss = self.compute_loss(X_val, val_reconstructed)
            
            # Record losses
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Track improvement
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if verbose:
                    print(f"\nEpoch {epoch + 1}: Validation loss improved by {improvement:.6f}")
            else:
                epochs_without_improvement += 1
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"\nEpoch {epoch + 1}: No improvement for {epochs_without_improvement} epochs")
                    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    print(f"Best validation loss so far: {best_val_loss:.4f}")
            
            # Early stopping check with detailed message
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("\n=== Early Stopping Triggered ===")
                print(f"Training stopped at epoch {epoch + 1}")
                print(f"Best validation loss achieved: {best_val_loss:.4f}")
                print(f"Final training loss: {train_loss:.4f}")
                print("==============================")
                break
                
            # Regular progress updates
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nProgress Update - Epoch {epoch + 1}/{epochs}")
                print(f"Current Metrics:")
                print(f"  - Training Loss: {train_loss:.4f}")
                print(f"  - Validation Loss: {val_loss:.4f}")
                print(f"  - Best Validation Loss: {best_val_loss:.4f}")
        
        # Final training summary
        print("\n=== Training Complete ===")
        print(f"Total epochs run: {epoch + 1}")
        print(f"Final training loss: {train_loss:.4f}")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=======================")
        
        return history


def main():
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv('butterfly_data.csv')

    # Define all features including population density
    features = [
        'Average Elevation (m)',
        'Average annual precipitiation (mm)',
        'Mean temp in C',
        'Extent of Forest (1000 ha)',
        'area (sq km)',
        'latitude',
        'island',
        'Population density (sq km)'  # New feature added
    ]

    # Clean and convert data
    print("\nCleaning data...")
    for feature in features:
        df[feature] = df[feature].astype(str).str.replace(',', '').astype(float)

    # Create and normalize feature matrix
    X = df[features].values
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # Create and train autoencoder
    input_dim = len(features)  # Now 8 features
    encoding_dim = 2
    autoencoder = SimpleAutoencoder(input_dim, encoding_dim, learning_rate=0.01)

    # Train the model
    print("\nStarting training...")
    # Update this line in your main() function where you call train
    history = autoencoder.train(X_normalized, epochs=100, batch_size=8, verbose=True, val_fraction=0.2)

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

    # Save results for visualization
    visualization_data = {
        'history': history,
        'features': features,
        'encoded_features': encoded_features,
        'reconstructed_data': reconstructed_data,
        'analysis_df': analysis_df,
        'species_richness': df['Number of species'].values
    }

    print("\nSaving data for visualization...")
    with open('autoencoder_results.pkl', 'wb') as f:
        pickle.dump(visualization_data, f)
    print("Data saved successfully!")

if __name__ == "__main__":
    main()