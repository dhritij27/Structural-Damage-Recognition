"""
Structural Damage Image Classification
ML Mini-Project - UE23CS352A

This module implements multiple machine learning models for binary classification
of structural images as damaged or undamaged.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
from pathlib import Path
import json

class StructuralDamageClassifier:
    """Main classifier class for structural damage detection"""
    
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load images and labels from directory structure"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Assuming directory structure: data_dir/damaged/ and data_dir/undamaged/
        for label, folder in enumerate(['undamaged', 'damaged']):
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                print(f"Warning: {folder_path} does not exist")
                continue
                
            for img_file in folder_path.glob('*.jpg'):
                try:
                    img = cv2.imread(str(img_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    # Normalize to [-1, 1]
                    img = (img / 128.0) - 1.0
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images")
        print(f"Damaged: {np.sum(labels)}, Undamaged: {len(labels) - np.sum(labels)}")
        
        # Split into train and validation (90/10)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            images, labels, test_size=0.1, random_state=42, stratify=labels
        )
        
        print(f"Training set: {len(self.X_train)} images")
        print(f"Validation set: {len(self.X_val)} images")
        
    def train_knn(self, k=5):
        """Train K-Nearest Neighbors classifier"""
        print(f"\nTraining KNN (k={k})...")
        
        # Flatten images for classical ML
        X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        X_val_flat = self.X_val.reshape(len(self.X_val), -1)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_flat, self.y_train)
        
        train_acc = knn.score(X_train_flat, self.y_train)
        val_acc = knn.score(X_val_flat, self.y_val)
        
        self.results['KNN'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': knn
        }
        
        print(f"KNN - Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}")
        return knn
    
    def train_logistic_regression(self):
        """Train Logistic Regression classifier"""
        print("\nTraining Logistic Regression...")
        
        X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        X_val_flat = self.X_val.reshape(len(self.X_val), -1)
        
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train_flat, self.y_train)
        
        train_acc = lr.score(X_train_flat, self.y_train)
        val_acc = lr.score(X_val_flat, self.y_val)
        
        self.results['Logistic_Regression'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': lr
        }
        
        print(f"LR - Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}")
        return lr
    
    def train_svm(self):
        """Train Support Vector Machine with RBF kernel"""
        print("\nTraining SVM with RBF kernel...")
        
        X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        X_val_flat = self.X_val.reshape(len(self.X_val), -1)
        
        svm = SVC(C=1.0, gamma=0.001, kernel='rbf', random_state=42)
        svm.fit(X_train_flat, self.y_train)
        
        train_acc = svm.score(X_train_flat, self.y_train)
        val_acc = svm.score(X_val_flat, self.y_val)
        
        self.results['SVM'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': svm
        }
        
        print(f"SVM - Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}")
        return svm
    
    def train_mobilenet(self, epochs=20, batch_size=32):
        """Train MobileNet with transfer learning"""
        print("\nTraining MobileNet with transfer learning...")
        
        # Load pre-trained MobileNet
        base_model = MobileNet(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('mobilenet_best.h5', save_best_only=True)
        ]
        
        history = model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        train_acc = model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)[1]
        
        self.results['MobileNet'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': model,
            'history': history.history
        }
        
        print(f"MobileNet - Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}")
        return model, history
    
    def train_inceptionv3(self, epochs=30, batch_size=32, unfreeze_layers=10):
        """Train InceptionV3 with transfer learning"""
        print("\nTraining InceptionV3 with transfer learning...")
        
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze most layers, unfreeze top layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=10,
            zoom_range=0.15,
            fill_mode='nearest'
        )
        
        callbacks = [
            EarlyStopping(patience=7, restore_best_weights=True),
            ModelCheckpoint('inceptionv3_best.h5', save_best_only=True)
        ]
        
        history = model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        train_acc = model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)[1]
        
        self.results['InceptionV3'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': model,
            'history': history.history
        }
        
        print(f"InceptionV3 - Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}")
        return model, history
    
    def evaluate_model(self, model_name):
        """Generate detailed evaluation metrics"""
        if model_name not in self.results:
            print(f"Model {model_name} not trained yet")
            return
        
        model = self.results[model_name]['model']
        
        if model_name in ['KNN', 'Logistic_Regression', 'SVM']:
            X_val_flat = self.X_val.reshape(len(self.X_val), -1)
            y_pred = model.predict(X_val_flat)
        else:
            y_pred_prob = model.predict(self.X_val)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(self.y_val, y_pred)
        report = classification_report(self.y_val, y_pred, 
                                      target_names=['Undamaged', 'Damaged'])
        
        print(f"\n=== {model_name} Evaluation ===")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        
        return cm, report
    
    def plot_training_history(self, model_name):
        """Plot training history for deep learning models"""
        if model_name not in self.results or 'history' not in self.results[model_name]:
            print(f"No training history for {model_name}")
            return
        
        history = self.results[model_name]['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"{'Model':<25} {'Train Acc':<15} {'Val Acc':<15}")
        print("-"*60)
        
        for model_name, result in self.results.items():
            train_acc = result['train_accuracy']
            val_acc = result['val_accuracy']
            print(f"{model_name:<25} {train_acc:<15.3f} {val_acc:<15.3f}")
        
        # Save results to JSON
        results_summary = {
            model: {
                'train_accuracy': float(data['train_accuracy']),
                'val_accuracy': float(data['val_accuracy'])
            }
            for model, data in self.results.items()
        }
        
        with open('model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print("\nResults saved to model_results.json")
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        for model_name, result in self.results.items():
            if model_name in ['MobileNet', 'InceptionV3']:
                result['model'].save(f'{model_name.lower()}_model.h5')
                print(f"Saved {model_name} to {model_name.lower()}_model.h5")


def main():
    """Main execution function"""
    print("="*60)
    print("STRUCTURAL DAMAGE IMAGE CLASSIFICATION")
    print("ML Mini-Project - UE23CS352A")
    print("="*60)
    
    # Initialize classifier
    classifier = StructuralDamageClassifier(data_dir='data')
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Train classical ML models
    print("\n" + "="*60)
    print("TRAINING CLASSICAL ML MODELS")
    print("="*60)
    classifier.train_knn(k=5)
    classifier.train_logistic_regression()
    classifier.train_svm()
    
    # Train deep learning models
    print("\n" + "="*60)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*60)
    classifier.train_mobilenet(epochs=20)
    classifier.train_inceptionv3(epochs=30)
    
    # Evaluate models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    for model_name in classifier.results.keys():
        classifier.evaluate_model(model_name)
    
    # Plot training histories
    classifier.plot_training_history('MobileNet')
    classifier.plot_training_history('InceptionV3')
    
    # Compare all models
    classifier.compare_models()
    
    # Save models
    classifier.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
