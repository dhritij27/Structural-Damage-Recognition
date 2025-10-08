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
    return cm


def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve'):
    """Plot ROC curve and calculate AUC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return roc_auc


def plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves'):
    """Plot learning curves to analyze bias/variance"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_misclassifications(X, y_true, y_pred, n_samples=12, 
                                 title="Misclassified Images"):
    """Visualize misclassified images"""
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    n_samples = min(n_samples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, n_samples, replace=False)
    
    rows = (n_samples + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, rows * 3.5))
    fig.suptitle(title, fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img_idx = selected_indices[idx]
            img = X[img_idx]
            # Denormalize
            img = (img + 1) / 2
            ax.imshow(img)
            
            true_label = "Damaged" if y_true[img_idx] == 1 else "Undamaged"
            pred_label = "Damaged" if y_pred[img_idx] == 1 else "Undamaged"
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                        fontsize=9, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_false_positives_negatives(X, y_true, y_pred, y_pred_proba=None):
    """Detailed analysis of false positives and false negatives"""
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    print(f"\nFalse Positives: {len(fp_indices)} images")
    print(f"False Negatives: {len(fn_indices)} images")
    
    if y_pred_proba is not None:
        print(f"\nFalse Positive Confidence: {np.mean(y_pred_proba[fp_indices]):.3f} ± {np.std(y_pred_proba[fp_indices]):.3f}")
        print(f"False Negative Confidence: {np.mean(1 - y_pred_proba[fn_indices]):.3f} ± {np.std(1 - y_pred_proba[fn_indices]):.3f}")
    
    # Visualize examples
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    fig.suptitle('False Positives (Top) vs False Negatives (Bottom)', fontsize=14)
    
    # False Positives
    if len(fp_indices) > 0:
        fp_samples = np.random.choice(fp_indices, min(6, len(fp_indices)), replace=False)
        for idx, ax in enumerate(axes[0]):
            if idx < len(fp_samples):
                img = (X[fp_samples[idx]] + 1) / 2
                ax.imshow(img)
                if y_pred_proba is not None:
                    conf = y_pred_proba[fp_samples[idx]]
                    ax.set_title(f'Conf: {conf:.2f}', fontsize=9)
            ax.axis('off')
    
    # False Negatives
    if len(fn_indices) > 0:
        fn_samples = np.random.choice(fn_indices, min(6, len(fn_indices)), replace=False)
        for idx, ax in enumerate(axes[1]):
            if idx < len(fn_samples):
                img = (X[fn_samples[idx]] + 1) / 2
                ax.imshow(img)
                if y_pred_proba is not None:
                    conf = y_pred_proba[fn_samples[idx]]
                    ax.set_title(f'Conf: {conf:.2f}', fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('false_positives_negatives_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_model_performance(results_dict):
    """Create comprehensive comparison visualization"""
    models = list(results_dict.keys())
    train_accs = [results_dict[m]['train_accuracy'] for m in models]
    val_accs = [results_dict[m]['val_accuracy'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart comparison
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax1.bar(x + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Overfitting analysis
    overfit_scores = [train_accs[i] - val_accs[i] for i in range(len(models))]
    colors = ['red' if s > 0.2 else 'orange' if s > 0.1 else 'green' for s in overfit_scores]
    
    ax2.barh(models, overfit_scores, color=colors, alpha=0.7)
    ax2.set_xlabel('Overfitting Score (Train - Val)', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=14)
    ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='High')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_data_augmentation_examples(img, n_examples=8):
    """Show examples of data augmentation"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.15,
        fill_mode='nearest'
    )
    
    img_batch = img.reshape((1,) + img.shape)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Data Augmentation Examples', fontsize=14)
    
    i = 0
    for batch in datagen.flow(img_batch, batch_size=1):
        aug_img = batch[0]
        aug_img = (aug_img + 1) / 2  # Denormalize
        ax = axes[i // 4, i % 4]
        ax.imshow(aug_img)
        ax.axis('off')
        i += 1
        if i >= n_examples:
            break
    
    plt.tight_layout()
    plt.savefig('data_augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_class_weights(y):
    """Calculate class weights for imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"\nClass Distribution:")
    print(f"Undamaged (0): {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Damaged (1): {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"\nCalculated Class Weights: {class_weight_dict}")
    
    return class_weight_dict


def save_predictions(X, y_true, y_pred, y_pred_proba, output_file='predictions.csv'):
    """Save predictions to CSV for further analysis"""
    import pandas as pd
    
    df = pd.DataFrame({
        'image_index': range(len(y_true)),
        'true_label': y_true,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_proba,
        'correct': y_true == y_pred
    })
    
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Correct predictions: {df['correct'].sum()} ({df['correct'].sum()/len(df)*100:.1f}%)")
    print(f"Incorrect predictions: {(~df['correct']).sum()} ({(~df['correct']).sum()/len(df)*100:.1f}%)")


def plot_prediction_confidence_distribution(y_pred_proba, y_true):
    """Plot distribution of prediction confidence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    ax1.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Prediction Confidence', fontsize=14)
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Separate by true class
    undamaged_probs = y_pred_proba[y_true == 0]
    damaged_probs = y_pred_proba[y_true == 1]
    
    ax2.hist(undamaged_probs, bins=50, alpha=0.5, label='True Undamaged', color='blue')
    ax2.hist(damaged_probs, bins=50, alpha=0.5, label='True Damaged', color='red')
    ax2.set_xlabel('Prediction Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Confidence by True Class', fontsize=14)
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Utility functions for Structural Damage Classification")
    print("Import these functions in your main script or notebooks")
.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
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
