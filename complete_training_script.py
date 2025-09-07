import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our custom classes (make sure the files are in the same directory)
from sign_language_preprocessing import SignLanguagePreprocessor
from sign_language_model import SignLanguageModel

class SignLanguageTrainer:
    def __init__(self, dataset_path, save_dir="sign_language_project"):
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.create_directories()
        
        # Initialize components
        self.preprocessor = SignLanguagePreprocessor(dataset_path)
        self.model_builder = SignLanguageModel()
        
        # Training parameters
        self.img_size = (64, 64)
        self.batch_size = 32
        self.epochs = 14
        self.learning_rate = 0.001
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "data"), exist_ok=True)
        
    def prepare_data(self, force_reload=False):
        """Prepare and load data"""
        data_path = os.path.join(self.save_dir, "data")
        
        if not force_reload and os.path.exists(os.path.join(data_path, "X_train.npy")):
            print("Loading preprocessed data...")
            return self.preprocessor.load_preprocessed_data(data_path)
        else:
            print("Preprocessing data...")
            # Load and preprocess data (limit samples for faster training on limited resources)
            X, y_categorical, y_raw = self.preprocessor.load_and_preprocess_data(
                max_samples_per_class=1500  # Reduced for faster training
            )
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
                X, y_categorical, test_size=0.2, val_size=0.1
            )
            
            # Save preprocessed data
            self.preprocessor.save_preprocessed_data(
                X_train, X_val, X_test, y_train, y_val, y_test, data_path
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test, self.preprocessor.classes
    
    def train_custom_cnn(self, X_train, X_val, y_train, y_val):
        """Train custom CNN model"""
        print("\n" + "="*50)
        print("TRAINING CUSTOM CNN MODEL")
        print("="*50)
        
        # Create and compile model
        model = self.model_builder.create_efficient_cnn()
        self.model_builder.compile_model(self.learning_rate)
        
        print(f"Model Architecture:")
        model.summary()
        
        # Train model
        model_path = os.path.join(self.save_dir, "models", "custom_cnn_model.keras")
        self.model_builder.get_callbacks(model_path)
        
        history = self.model_builder.train_model(
            X_train, y_train, X_val, y_val, 
            epochs=self.epochs, batch_size=self.batch_size
        )
        
        return history, "custom_cnn"
    
    def train_transfer_learning(self, X_train, X_val, y_train, y_val):
        """Train transfer learning model"""
        print("\n" + "="*50)
        print("TRAINING TRANSFER LEARNING MODEL (MobileNetV2)")
        print("="*50)
        
        # Create and compile model
        model = self.model_builder.create_transfer_learning_model()
        self.model_builder.compile_model(self.learning_rate)
        
        print(f"Model Architecture:")
        model.summary()
        
        # Train model
        model_path = os.path.join(self.save_dir, "models", "transfer_learning_model.keras")
        self.model_builder.get_callbacks(model_path)
        
        history = self.model_builder.train_model(
            X_train, y_train, X_val, y_val, 
            epochs=self.epochs//2,  # Transfer learning needs fewer epochs
            batch_size=self.batch_size
        )
        
        return history, "transfer_learning"
    
    def evaluate_and_analyze(self, X_test, y_test, classes, model_type):
        """Comprehensive model evaluation"""
        print(f"\n" + "="*50)
        print(f"EVALUATING {model_type.upper()} MODEL")
        print("="*50)
        
        # Basic evaluation
        test_loss, test_accuracy, test_top3_accuracy = self.model_builder.evaluate_model(X_test, y_test)
        
        # Detailed predictions
        y_pred_proba = self.model_builder.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=classes))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, classes, model_type)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, classes, model_type):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_type.title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "plots", f"confusion_matrix_{model_type}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def compare_models(self, results):
        """Compare different model results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_type, result in results.items():
            comparison_data.append([
                model_type.title(),
                f"{result['test_accuracy']:.4f}",
                f"{result['test_top3_accuracy']:.4f}",
                f"{result['test_loss']:.4f}"
            ])
        
        print(f"{'Model':<20} {'Accuracy':<12} {'Top-3 Acc':<12} {'Loss':<10}")
        print("-" * 60)
        for row in comparison_data:
            print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<10}")
    
    def save_training_summary(self, results):
        """Save training summary"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_path = os.path.join(self.save_dir, f"training_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Sign Language Recognition Training Summary\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Image Size: {self.img_size}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n\n")
            
            for model_type, result in results.items():
                f.write(f"\n{model_type.upper()} MODEL RESULTS:\n")
                f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write(f"Test Top-3 Accuracy: {result['test_top3_accuracy']:.4f}\n")
                f.write(f"Test Loss: {result['test_loss']:.4f}\n")
        
        print(f"Training summary saved to: {summary_path}")
    
    def create_deployment_files(self, classes):
        """Create files needed for deployment"""
        # Save classes for real-time detection
        import pickle
        classes_path = os.path.join(self.save_dir, "classes.pkl")
        with open(classes_path, 'wb') as f:
            pickle.dump(classes, f)
        
        # Also save classes with model-specific names for easier loading
        for model_type in results.keys():
            if model_type == "custom_cnn":
                model_classes_path = os.path.join(self.save_dir, "models", "custom_cnn_model_classes.pkl")
            elif model_type == "transfer_learning":
                model_classes_path = os.path.join(self.save_dir, "models", "transfer_learning_model_classes.pkl")
            else:
                model_classes_path = os.path.join(self.save_dir, "models", f"{model_type}_classes.pkl")
            
            with open(model_classes_path, 'wb') as f:
                pickle.dump(classes, f)
        
        # Save a general classes file for easier loading
        general_classes_path = os.path.join(self.save_dir, "models", "classes.pkl")
        with open(general_classes_path, 'wb') as f:
            pickle.dump(classes, f)
        
        # Create requirements.txt
        requirements_path = os.path.join(self.save_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            requirements = [
                "tensorflow>=2.8.0",
                "opencv-python>=4.5.0",
                "mediapipe>=0.8.0",
                "numpy>=1.21.0",
                "matplotlib>=3.3.0",
                "scikit-learn>=1.0.0",
                "pyttsx3>=2.90",
                "seaborn>=0.11.0"
            ]
            f.write('\n'.join(requirements))
        
        print(f"Classes saved to: {classes_path}")
        print(f"Requirements saved to: {requirements_path}")
    
    def run_complete_training(self, train_both_models=True):
        """Run complete training pipeline"""
        print(" Starting Sign Language Recognition Training Pipeline")
        print("="*60)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, classes = self.prepare_data()
        
        print(f"\nDataset Information:")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Number of classes: {len(classes)}")
        print(f"Image shape: {X_train.shape[1:]}")
        
        results = {}
        
        if train_both_models:
            # Train both models
            try:
                # Custom CNN
                history_cnn, model_type_cnn = self.train_custom_cnn(X_train, X_val, y_train, y_val)
                self.model_builder.plot_training_history(history_cnn)
                results[model_type_cnn] = self.evaluate_and_analyze(X_test, y_test, classes, model_type_cnn)
                
                # Transfer Learning
                history_tl, model_type_tl = self.train_transfer_learning(X_train, X_val, y_train, y_val)
                self.model_builder.plot_training_history(history_tl)
                results[model_type_tl] = self.evaluate_and_analyze(X_test, y_test, classes, model_type_tl)
                
                # Compare models
                self.compare_models(results)
                
            except Exception as e:
                print(f"Error during training: {e}")
                print("Training with custom CNN only...")
                
                # Train custom CNN only
                history_cnn, model_type_cnn = self.train_custom_cnn(X_train, X_val, y_train, y_val)
                self.model_builder.plot_training_history(history_cnn)
                results[model_type_cnn] = self.evaluate_and_analyze(X_test, y_test, classes, model_type_cnn)
        else:
            # Train custom CNN only
            history_cnn, model_type_cnn = self.train_custom_cnn(X_train, X_val, y_train, y_val)
            self.model_builder.plot_training_history(history_cnn)
            results[model_type_cnn] = self.evaluate_and_analyze(X_test, y_test, classes, model_type_cnn)
        
        # Save summary and create deployment files
        self.save_training_summary(results)
        self.create_deployment_files(classes)
        
        print("\n Training completed successfully!")
        print(f" All files saved in: {self.save_dir}")
        print("\nNext steps:")
        print("1. Use the best performing model for real-time detection")
        print("2. Run the real-time detection script")
        print("3. Test with different lighting conditions and backgrounds")
        
        return results

# Main execution
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "datasetasl/asl_alphabet_train/asl_alphabet_train"  # Path to your ASL dataset
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(" Error: Dataset path not found!")
        print(f"Please check if the dataset exists at: {DATASET_PATH}")
        print("Expected structure: DATASET_PATH/A/, DATASET_PATH/B/, etc.")
        print("Make sure the dataset folder contains subfolders for each letter (A, B, C, etc.)")
        sys.exit(1)
    
    try:
        # Initialize trainer
        trainer = SignLanguageTrainer(DATASET_PATH)
        
        # Run complete training
        results = trainer.run_complete_training(train_both_models=True)
        
        # Print final recommendations
        print("\n RECOMMENDATIONS FOR IMPRESSIVE RESULTS:")
        print("1. Use data augmentation to improve generalization")
        print("2. Test the model in different lighting conditions")
        print("3. Add background subtraction for better hand isolation")
        print("4. Implement gesture smoothing for more stable predictions")
        print("5. Add word completion and spell-checking features")
        print("6. Create a user-friendly GUI interface")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    except Exception as e:
        print(f" Error during training: {e}")
        print("Please check your dataset path and requirements")