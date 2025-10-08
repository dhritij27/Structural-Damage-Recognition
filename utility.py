"""
Utility functions for structural damage classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import cv2
from pathlib import Path


def visualize_samples(X, y, n_samples=16, title="Sample Images"):
    """Visualize random samples from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        img = X[indices[idx]]
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        ax.imshow(img)
        label = "Damaged" if y[indices[idx]] == 1 else "Undamaged"
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=['Undamaged', 'Damaged'], 
                         title='Confusion Matrix'):
    """Plot confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt
