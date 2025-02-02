import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_and_verify_data(filepath='autoencoder_results.pkl'):
    """Load data and print key information for verification."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Print data structure for verification
    print("\nData structure verification:")
    print(f"Number of features: {len(data['features'])}")
    print(f"Encoded features shape: {data['encoded_features'].shape}")
    print(f"Features available: {data['features']}")
    
    return data

def create_correlation_matrix(data):
    """Generate correlation matrix between original features and encoded dimensions."""
    # Initialize correlation matrix
    correlations = np.zeros((len(data['features']), 2))
    
    # Calculate correlations for each feature with both encoded dimensions
    for i, feature in enumerate(data['features']):
        original_feature = data['analysis_df'][feature].values
        correlations[i, 0] = np.corrcoef(original_feature, data['encoded_features'][:, 0])[0, 1]
        correlations[i, 1] = np.corrcoef(original_feature, data['encoded_features'][:, 1])[0, 1]
        
        # Print correlations for verification
        print(f"\n{feature}:")
        print(f"  Dimension 1: {correlations[i, 0]:.4f}")
        print(f"  Dimension 2: {correlations[i, 1]:.4f}")
    
    return correlations

def plot_correlation_matrix(correlations, features):
    """Create and save correlation matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(correlations, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(2):
            plt.text(j, i, f'{correlations[i,j]:.2f}', 
                    ha='center', va='center',
                    color='black')
    
    # Labels and formatting
    plt.xticks([0, 1], ['Dimension 1', 'Dimension 2'])
    plt.yticks(range(len(features)), features)
    plt.title('Feature Correlations with Encoded Dimensions')
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(data):
    """Visualize training dynamics with confidence bands."""
    train_loss = data['history']['train_loss']
    val_loss = data['history']['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot main loss curves
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', alpha=0.7)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(epochs, train_loss, 1)
    trend = np.poly1d(z)
    plt.plot(epochs, trend(epochs), 'k--', 
             label=f'Trend (slope: {z[0]:.2e})', alpha=0.5)
    
    plt.title('Autoencoder Training Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space(data):
    """Create a simplified visualization of the latent space focusing on key patterns."""
    encoded_features = data['encoded_features']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Species richness with island status
    scatter1 = ax1.scatter(encoded_features[:,0], encoded_features[:,1],
                          c=data['species_richness'], cmap='viridis',
                          alpha=0.7)
    # Add island/mainland markers
    is_island = data['analysis_df']['island'] == 1
    ax1.scatter(encoded_features[is_island,0], encoded_features[is_island,1],
                facecolors='none', edgecolors='red', alpha=0.5)
    
    ax1.set_title('Species Richness Distribution\n(Red circles: Islands)')
    ax1.set_xlabel('Dimension 1 (Island-Mainland)')
    ax1.set_ylabel('Dimension 2 (Climate)')
    plt.colorbar(scatter1, ax=ax1, label='Number of Species')
    
    # 2. Temperature gradient
    scatter2 = ax2.scatter(encoded_features[:,0], encoded_features[:,1],
                          c=data['analysis_df']['Mean temp in C'],
                          cmap='RdBu_r', alpha=0.7)
    ax2.set_title('Temperature Distribution')
    ax2.set_xlabel('Dimension 1 (Island-Mainland)')
    ax2.set_ylabel('Dimension 2 (Climate)')
    plt.colorbar(scatter2, ax=ax2, label='Mean Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('latent_space_simplified.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_quality(data):
    """Visualize reconstruction quality for key ecological features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate reconstruction error for each feature
    features = data['features']
    reconstructed = data['reconstructed_data']
    original = data['analysis_df'][features].values
    
    errors = np.mean((original - reconstructed)**2, axis=0)
    
    # Sort features by reconstruction error
    sorted_idx = np.argsort(errors)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_errors = errors[sorted_idx]
    
    # Create bar plot
    bars = ax.barh(range(len(features)), sorted_errors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Mean Squared Reconstruction Error')
    ax.set_title('Feature Reconstruction Quality')
    
    plt.tight_layout()
    plt.savefig('reconstruction_quality.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        print("Loading autoencoder results...")
        data = load_and_verify_data()
        
        print("\nGenerating visualizations...")
        
        # Create correlation matrix
        print("1. Calculating feature correlations...")
        correlations = create_correlation_matrix(data)
        plot_correlation_matrix(correlations, data['features'])
        
        # Add training dynamics plot
        print("2. Creating training dynamics visualization...")
        plot_training_history(data)

        plot_latent_space(data)

        plot_reconstruction_quality(data)
        
        print("\nVisualization complete! Generated files:")
        print("- correlation_matrix.png")
        print("- training_dynamics.png")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please verify data structure and file availability.")

if __name__ == "__main__":
    main()