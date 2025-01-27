import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd

def load_data(filepath='autoencoder_results.pkl'):
    """Load the saved autoencoder results."""
    print("Loading data from pickle file...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("Data loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found!")
        return None

def validate_data(data):
    """Validate that all required data is present and has the expected structure."""
    required_keys = ['history', 'features', 'encoded_features', 
                    'reconstructed_data', 'analysis_df', 'species_richness']
    
    print("\nValidating data structure...")
    
    # Check all keys exist
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key '{key}'")
            return False
            
    # Validate shapes and types
    print("\nData shapes and types:")
    print(f"Training history length: {len(data['history'])}")
    print(f"Number of features: {len(data['features'])}")
    print(f"Encoded features shape: {data['encoded_features'].shape}")
    print(f"Reconstructed data shape: {data['reconstructed_data'].shape}")
    print(f"Analysis DataFrame shape: {data['analysis_df'].shape}")
    print(f"Species richness length: {len(data['species_richness'])}")
    
    return True

def create_paper_visualizations(data):
    """Create all visualizations for the paper."""
    
    print("\nCreating visualizations...")
    
    # 1. Training Loss Curve
    print("Creating loss curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data['history']) + 1), data['history'], 'b-')
    plt.title('Autoencoder Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Loss curve saved as 'loss_curve.png'")

    # 2. Feature-Dimension Correlation Heatmap
    print("\nCreating correlation heatmap...")
    correlations = []
    for feature in data['features']:
        corr_dim1 = data['analysis_df'][feature].corr(data['analysis_df']['encoded_dim_1'])
        corr_dim2 = data['analysis_df'][feature].corr(data['analysis_df']['encoded_dim_2'])
        correlations.append([corr_dim1, corr_dim2])

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, 
                annot=True, 
                cmap='RdBu', 
                center=0,
                xticklabels=['Dimension 1', 'Dimension 2'],
                yticklabels=data['features'])
    plt.title('Feature Correlations with Encoded Dimensions')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")

    # 3. Species Richness vs. Encoded Dimensions
    print("\nCreating species richness scatter plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dimension 1
    ax1.scatter(data['analysis_df']['encoded_dim_1'], data['species_richness'])
    z1 = np.polyfit(data['analysis_df']['encoded_dim_1'], data['species_richness'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(data['analysis_df']['encoded_dim_1'], 
             p1(data['analysis_df']['encoded_dim_1']), "r--", alpha=0.8)
    ax1.set_xlabel('Encoded Dimension 1')
    ax1.set_ylabel('Species Richness')
    ax1.set_title('Species Richness vs. Dimension 1')
    
    # Dimension 2
    ax2.scatter(data['analysis_df']['encoded_dim_2'], data['species_richness'])
    z2 = np.polyfit(data['analysis_df']['encoded_dim_2'], data['species_richness'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(data['analysis_df']['encoded_dim_2'], 
             p2(data['analysis_df']['encoded_dim_2']), "r--", alpha=0.8)
    ax2.set_xlabel('Encoded Dimension 2')
    ax2.set_ylabel('Species Richness')
    ax2.set_title('Species Richness vs. Dimension 2')
    
    plt.tight_layout()
    plt.savefig('richness_vs_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Species richness plots saved as 'richness_vs_dimensions.png'")

    # 4. 2D Embedding Plot
    print("\nCreating 2D embedding plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(data['encoded_features'][:, 0], 
                         data['encoded_features'][:, 1],
                         c=data['species_richness'],
                         s=data['analysis_df']['area (sq km)'] / data['analysis_df']['area (sq km)'].max() * 500,
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='Species Richness')
    plt.xlabel('Encoded Dimension 1')
    plt.ylabel('Encoded Dimension 2')
    plt.title('2D Encoded Space: Regions Colored by Species Richness, Sized by Area')
    
    plt.tight_layout()
    plt.savefig('2d_embedding.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("2D embedding plot saved as '2d_embedding.png'")

    # 5. Reconstruction Error Bar Plot
    print("\nCreating reconstruction error plot...")
    reconstruction_errors = []
    for feature in data['features']:
        mse = np.mean((data['analysis_df'][feature] - 
                      pd.DataFrame(data['reconstructed_data'], 
                                 columns=data['features'])[feature]) ** 2)
        reconstruction_errors.append(mse)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(data['features'], reconstruction_errors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Reconstruction Error by Feature')
    plt.ylabel('Mean Squared Error')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reconstruction_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Reconstruction error plot saved as 'reconstruction_error.png'")

    print("\nAll visualizations completed successfully!")

def main():
    # Load the data
    data = load_data()
    if data is None:
        return
    
    # Validate the data
    if not validate_data(data):
        print("Error: Data validation failed!")
        return
    
    # Create visualizations
    create_paper_visualizations(data)

if __name__ == "__main__":
    main()