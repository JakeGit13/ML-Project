import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

def load_data(filepath='autoencoder_results.pkl'):
    """Load the autoencoder results and verify data structure."""
    print("Loading data from pickle file...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Verify expected keys are present
    expected_keys = ['history', 'features', 'encoded_features', 
                    'reconstructed_data', 'species_richness']
    missing_keys = [key for key in expected_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing expected keys: {missing_keys}")
    
    print("Data loaded successfully!")
    return data

def create_visualizations(data):
    """Create comprehensive visualizations using minimal dependencies."""
    plt.style.use('default')
    
    # 1. Training Loss Evolution
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(data['history']) + 1)
    plt.plot(epochs, data['history'], 'b-', linewidth=2, label='Training Loss')
    
    # Add trend line
    z = np.polyfit(epochs, data['history'], 1)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), "r--", alpha=0.8, 
             label=f'Trend (slope: {z[0]:.4f})')
    
    plt.title('Autoencoder Training Progress', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('loss_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

   # Replace the existing 2D visualization section with this:

    # 3. Integrated 2D Visualization
    plt.figure(figsize=(12, 8))
    
    # Create base scatter plot (always works with species richness)
    scatter = plt.scatter(data['encoded_features'][:, 0], 
                         data['encoded_features'][:, 1],
                         c=data['species_richness'],
                         cmap='viridis',
                         alpha=0.7)
    
    plt.colorbar(scatter, label='Species Richness')
    
    # Add population density encoding if available
    density_cols = [col for col in data['features'] if 'density' in col.lower()]
    if density_cols:
        # Update point sizes based on population density
        pop_density_idx = data['features'].index(density_cols[0])
        pop_density = data['reconstructed_data'][:, pop_density_idx]
        plt.scatter(data['encoded_features'][:, 0], 
                   data['encoded_features'][:, 1],
                   c=data['species_richness'],
                   s=np.log1p(pop_density) * 100,
                   cmap='viridis',
                   alpha=0.7)
        
        # Add annotation for highest density
        max_density_idx = np.argmax(pop_density)
        plt.annotate('Highest Population Density',
                    xy=(data['encoded_features'][max_density_idx, 0],
                        data['encoded_features'][max_density_idx, 1]),
                    xytext=(30, 30),
                    textcoords='offset points',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->',
                                   connectionstyle='arc3,rad=0.2'))
    
    # Always add highest species richness annotation
    max_richness_idx = np.argmax(data['species_richness'])
    plt.annotate('Highest Species Richness',
                xy=(data['encoded_features'][max_richness_idx, 0],
                    data['encoded_features'][max_richness_idx, 1]),
                xytext=(-30, -30),
                textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='->',
                               connectionstyle='arc3,rad=-0.2'))
    
    plt.xlabel('First Encoded Dimension', fontsize=10)
    plt.ylabel('Second Encoded Dimension', fontsize=10)
    plt.title('Ecological Patterns in Reduced Dimensional Space\n' +
             'Color: Species Richness',
             fontsize=12, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_embedding.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature Reconstruction Analysis
    reconstructed_df = pd.DataFrame(data['reconstructed_data'], 
                                  columns=data['features'])
    
    errors = []
    for feature in data['features']:
        error = np.mean((reconstructed_df[feature] - 
                        reconstructed_df[feature].mean()) ** 2)
        errors.append({'feature': feature, 'error': error})
    
    error_df = pd.DataFrame(errors)
    error_df = error_df.sort_values('error', ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(error_df)), error_df['error'], alpha=0.8)
    
    plt.xticks(range(len(error_df)), 
               error_df['feature'], 
               rotation=45, 
               ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=8)
    
    plt.title('Feature Reconstruction Analysis', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reconstruction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Correlation Matrix
    plt.figure(figsize=(10, 8))
    
    # Calculate correlations between encoded dimensions and features
    correlations = []
    for feature in data['features']:
        feature_data = reconstructed_df[feature]
        corr_dim1 = np.corrcoef(data['encoded_features'][:, 0], feature_data)[0,1]
        corr_dim2 = np.corrcoef(data['encoded_features'][:, 1], feature_data)[0,1]
        correlations.append([corr_dim1, corr_dim2])
    
    correlations = np.array(correlations)
    
    # Create correlation matrix plot
    im = plt.imshow(correlations, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    
    # Add correlation values as text
    for i in range(len(data['features'])):
        for j in range(2):
            plt.text(j, i, f'{correlations[i,j]:.2f}', 
                    ha='center', va='center', color='black')
    
    plt.xticks([0, 1], ['Dimension 1', 'Dimension 2'])
    plt.yticks(range(len(data['features'])), data['features'], ha='right')
    plt.title('Feature Correlations with Encoded Dimensions', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        print("\nStarting visualization process...")
        data = load_data()
        
        print("\nCreating visualizations...")
        create_visualizations(data)
        
        print("\nVisualizations created successfully!")
        print("Generated files:")
        print("- loss_evolution.png")
        print("- population_patterns.png")
        print("- integrated_patterns.png")
        print("- reconstruction_analysis.png")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check your data structure and column names.")