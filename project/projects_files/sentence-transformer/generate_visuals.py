"""
Script to generate visualizations for the sentence transformer project write-up.

This script can be run after training or with example data to create
the visualizations referenced in the write-up.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import torch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create assets directory
assets_dir = Path('assets')
assets_dir.mkdir(exist_ok=True)

def generate_embedding_heatmap(embeddings, save_path='assets/embedding_heatmap.png'):
    """Generate heatmap visualization of a single sentence embedding."""
    if isinstance(embeddings, torch.Tensor):
        embedding_np = embeddings[0].cpu().numpy()
    else:
        embedding_np = np.array(embeddings[0])
    
    plt.figure(figsize=(14, 2))
    plt.imshow(embedding_np.reshape(1, -1), cmap="coolwarm", aspect="auto")
    plt.colorbar(label='Embedding Value')
    plt.title("Sentence Embedding Visualization (1024 dimensions)", fontsize=14, fontweight='bold')
    plt.xlabel("Embedding Dimensions", fontsize=12)
    plt.ylabel("Sentence", fontsize=12)
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_similarity_matrix(embeddings, sentences=None, save_path='assets/similarity_matrix.png'):
    """Generate cosine similarity matrix heatmap."""
    from sentence_transformers.util import cos_sim
    
    if isinstance(embeddings, torch.Tensor):
        similarity_matrix = cos_sim(embeddings, embeddings).cpu().numpy()
    else:
        similarity_matrix = cos_sim(embeddings, embeddings).numpy()
    
    num_sentences = len(similarity_matrix)
    num_labels = [str(i+1) for i in range(num_sentences)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        vmin=0, 
        vmax=1, 
        fmt=".2f", 
        cmap="coolwarm", 
        xticklabels=num_labels, 
        yticklabels=num_labels, 
        ax=ax,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    ax.set_xticklabels(num_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(num_labels, fontsize=10)
    ax.set_title("Cosine Similarity Matrix of Sentence Embeddings", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Sentence Index", fontsize=12)
    ax.set_ylabel("Sentence Index", fontsize=12)
    
    if sentences:
        sentence_text = "\n".join([f"{i+1}. {sent[:50]}..." if len(sent) > 50 else f"{i+1}. {sent}" 
                                  for i, sent in enumerate(sentences)])
        props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
        fig.text(0.02, 0.98, sentence_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_tsne_clustering(embeddings, sentences=None, n_clusters=4, 
                            save_path='assets/tsne_clustering.png'):
    """Generate t-SNE visualization with K-means clustering."""
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings_np)-1), 
                random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Colors for clusters
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    cluster_colors = [colors[label % len(colors)] for label in labels]
    
    plt.figure(figsize=(8, 8))
    for i in range(len(embeddings_2d)):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                   color=cluster_colors[i], edgecolors='k', s=150, alpha=0.7)
        if sentences:
            label = f"{i+1}"
            plt.text(embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1] + 0.5, 
                    label, fontsize=9, alpha=0.8)
    
    plt.title("t-SNE Visualization of Sentence Clusters", 
             fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Legend
    legend_labels = [f"Cluster {i}" for i in range(n_clusters)]
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=colors[i], markersize=12, 
                          markeredgecolor='k') for i in range(n_clusters)]
    plt.legend(handles, legend_labels, title="Clusters", loc="best", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_training_curves(train_class_losses, val_class_losses,
                            train_ner_losses, val_ner_losses,
                            save_path='assets/training_curves.png'):
    """Generate training and validation loss curves."""
    epochs = range(1, len(train_class_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classification loss
    ax1.plot(epochs, train_class_losses, label='Train', marker='o', linewidth=2, markersize=6)
    ax1.plot(epochs, val_class_losses, label='Validation', marker='s', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Classification Loss', fontsize=12)
    ax1.set_title('Classification Loss Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # NER loss
    ax2.plot(epochs, train_ner_losses, label='Train', marker='o', linewidth=2, markersize=6)
    ax2.plot(epochs, val_ner_losses, label='Validation', marker='s', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('NER Loss', fontsize=12)
    ax2.set_title('NER Loss Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_accuracy_curves(train_class_acc, val_class_acc,
                            train_ner_acc=None, val_ner_acc=None,
                            save_path='assets/accuracy_curves.png'):
    """Generate accuracy curves for classification and optionally NER."""
    epochs = range(1, len(train_class_acc) + 1)
    
    if train_ner_acc is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax2 = None
    
    # Classification accuracy
    ax1.plot(epochs, train_class_acc, label='Train', marker='o', linewidth=2, markersize=6)
    ax1.plot(epochs, val_class_acc, label='Validation', marker='s', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # NER accuracy (if provided)
    if ax2 is not None and train_ner_acc is not None:
        ax2.plot(epochs, train_ner_acc, label='Train', marker='o', linewidth=2, markersize=6)
        ax2.plot(epochs, val_ner_acc, label='Validation', marker='s', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('NER Accuracy', fontsize=12)
        ax2.set_title('NER Accuracy Over Time', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_example_prediction(sentence, classification, ner_tags, 
                                save_path='assets/example_prediction.png'):
    """Generate visualization of a single prediction with classification and NER."""
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Tokenize sentence (simple whitespace tokenization for visualization)
    tokens = sentence.split()
    
    # Create color map for NER tags
    tag_colors = {
        'O': '#ffffff',
        'B-PRODUCT': '#ffcccc',
        'I-PRODUCT': '#ffcccc',
        'B-PRICE': '#ccffcc',
        'I-PRICE': '#ccffcc'
    }
    
    # Plot tokens with background colors
    y_pos = 0.5
    x_start = 0.1
    x_pos = x_start
    
    for i, (token, tag) in enumerate(zip(tokens, ner_tags[:len(tokens)])):
        color = tag_colors.get(tag, '#ffffff')
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='gray', alpha=0.7)
        ax.text(x_pos, y_pos, token, fontsize=11, bbox=bbox)
        x_pos += len(token) * 0.08 + 0.15
    
    # Add classification label
    ax.text(0.5, 0.9, f'Classification: {classification}', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Example Prediction', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Visual generation script for sentence transformer project")
    print("=" * 60)
    print("\nTo use this script:")
    print("1. Load your embeddings or model outputs")
    print("2. Call the appropriate generation functions")
    print("3. Visuals will be saved to the assets/ directory")
    print("\nExample usage:")
    print("  from generate_visuals import *")
    print("  # Load embeddings from your notebook")
    print("  generate_embedding_heatmap(embeddings)")
    print("  generate_similarity_matrix(embeddings, sentences)")
