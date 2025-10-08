"""
Evaluation script for trained models
"""

import numpy as np
import argparse
from tensorflow.keras.models import load_model
from pathlib import Path
import json
from main import StructuralDamageClassifier
from utils import (
    visualize_samples, plot_confusion_matrix, plot_roc_curve,
    visualize_misclassifications, analyze_false_positives_negatives,
    plot_prediction_confidence_distribution, save_predictions
)


def evaluate_saved_model(model_path, data_dir='data', visualize=True):
    """Evaluate a saved deep learning model"""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load data
    classifier = StructuralDamageClassifier(data_dir=data_dir)
    classifier.load_and_preprocess_data()
    
    # Get predictions
    y_pred_proba = model.predict(classifier.X_val).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == classifier.y_val)
    print(f"\nValidation Accuracy: {accuracy:.3f}")
    
    if visualize:
        # Confusion matrix
        plot_confusion_matrix(
            classifier.y_val, y_pred,
            title=f'{Path(model_path).stem} - Confusion Matrix'
        )
        
        # ROC curve
        roc_auc = plot_roc_curve(
            classifier.y_val, y_pred_proba,
            title=f'{Path(model_path).stem} - ROC Curve'
        )
        print(f"AUC Score: {roc_auc:.3f}")
        
        # Misclassifications
        visualize_misclassifications(
            classifier.X_val, classifier.y_val, y_pred,
            title=f'{Path(model_path).stem} - Misclassifications'
        )
        
        # False positive/negative analysis
        analyze_false_positives_negatives(
            classifier.X_val, classifier.y_val, y_pred, y_pred_proba
        )
        
        # Prediction confidence
        plot_prediction_confidence_distribution(y_pred_proba, classifier.y_val)
        
        # Save predictions
        save_predictions(
            classifier.X_val, classifier.y_val, y_pred, y_pred_proba,
            output_file=f'{Path(model_path).stem}_predictions.csv'
        )
    
    return accuracy, y_pred, y_pred_proba


def compare_all_models(results_file='model_results.json'):
    """Load and compare all model results"""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found. Please run main.py first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'Train Acc':<15} {'Val Acc':<15} {'Overfit':<10}")
    print("-"*70)
    
    for model_name, metrics in results.items():
        train_acc = metrics['train_accuracy']
        val_acc = metrics['val_accuracy']
        overfit = train_acc - val_acc
        
        print(f"{model_name:<30} {train_acc:<15.3f} {val_acc:<15.3f} {overfit:<10.3f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['val_accuracy'])
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model[0]}")
    print(f"Validation Accuracy: {best_model[1]['val_accuracy']:.3f}")
    print("="*70)


def visualize_dataset_samples(data_dir='data'):
    """Visualize random samples from the dataset"""
    classifier = StructuralDamageClassifier(data_dir=data_dir)
    classifier.load_and_preprocess_data()
    
    # Training samples
    visualize_samples(
        classifier.X_train, classifier.y_train,
        n_samples=16, title="Training Set Samples"
    )
    
    # Validation samples
    visualize_samples(
        classifier.X_val, classifier.y_val,
        n_samples=16, title="Validation Set Samples"
    )
    
    # Class distribution
    print("\nDataset Statistics:")
    print(f"Training set: {len(classifier.X_train)} images")
    print(f"  - Undamaged: {np.sum(classifier.y_train == 0)} ({np.sum(classifier.y_train == 0)/len(classifier.y_train)*100:.1f}%)")
    print(f"  - Damaged: {np.sum(classifier.y_train == 1)} ({np.sum(classifier.y_train == 1)/len(classifier.y_train)*100:.1f}%)")
    print(f"\nValidation set: {len(classifier.X_val)} images")
    print(f"  - Undamaged: {np.sum(classifier.y_val == 0)} ({np.sum(classifier.y_val == 0)/len(classifier.y_val)*100:.1f}%)")
    print(f"  - Damaged: {np.sum(classifier.y_val == 1)} ({np.sum(classifier.y_val == 1)/len(classifier.y_val)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate structural damage classification models')
    parser.add_argument('--model', type=str, choices=['mobilenet', 'inceptionv3', 'all'],
                       default='all', help='Model to evaluate')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualization plots')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models')
    parser.add_argument('--samples', action='store_true',
                       help='Visualize dataset samples')
    
    args = parser.parse_args()
    
    if args.samples:
        visualize_dataset_samples(args.data_dir)
        return
    
    if args.compare:
        compare_all_models()
        return
    
    if args.model == 'all':
        print("Evaluating all models...")
        
        if Path('mobilenet_model.h5').exists():
            print("\n" + "="*70)
            print("MOBILENET EVALUATION")
            print("="*70)
            evaluate_saved_model('mobilenet_model.h5', args.data_dir, args.visualize)
        
        if Path('inceptionv3_model.h5').exists():
            print("\n" + "="*70)
            print("INCEPTIONV3 EVALUATION")
            print("="*70)
            evaluate_saved_model('inceptionv3_model.h5', args.data_dir, args.visualize)
        
        compare_all_models()
    
    else:
        model_path = f'{args.model}_model.h5'
        if not Path(model_path).exists():
            print(f"Model file {model_path} not found. Please train the model first.")
            return
        
        evaluate_saved_model(model_path, args.data_dir, args.visualize)


if __name__ == "__main__":
    main()
