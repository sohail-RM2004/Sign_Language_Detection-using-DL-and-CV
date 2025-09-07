import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pickle

class SignLanguagePreprocessor:
    def __init__(self, dataset_path, img_size=(64, 64)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                       'space', 'del', 'nothing']
        self.num_classes = len(self.classes)
        
    def load_and_preprocess_data(self, max_samples_per_class=2000):
        """
        Load and preprocess the dataset with memory optimization
        max_samples_per_class: Limit samples to reduce memory usage
        """
        X = []
        y = []
        
        print("Loading dataset...")
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class {class_name} not found in dataset")
                continue
                
            image_files = os.listdir(class_path)[:max_samples_per_class]
            
            for i, img_file in enumerate(image_files):
                if i % 200 == 0:
                    print(f"Processing {class_name}: {i}/{len(image_files)}")
                
                img_path = os.path.join(class_path, img_file)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                # Normalize pixel values
                img = img.astype('float32') / 255.0
                
                X.append(img)
                y.append(class_idx)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle data
        X, y = shuffle(X, y, random_state=42)
        
        # One-hot encode labels
        y_categorical = to_categorical(y, self.num_classes)
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1:]} image shape")
        
        return X, y_categorical, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Data split:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                               save_path="preprocessed_data"):
        """Save preprocessed data to avoid reprocessing"""
        os.makedirs(save_path, exist_ok=True)
        
        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_val.npy'), X_val)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_val.npy'), y_val)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        
        # Save class names
        with open(os.path.join(save_path, 'classes.pkl'), 'wb') as f:
            pickle.dump(self.classes, f)
            
        print(f"Preprocessed data saved to {save_path}")
    
    def load_preprocessed_data(self, load_path="preprocessed_data"):
        """Load preprocessed data"""
        X_train = np.load(os.path.join(load_path, 'X_train.npy'))
        X_val = np.load(os.path.join(load_path, 'X_val.npy'))
        X_test = np.load(os.path.join(load_path, 'X_test.npy'))
        y_train = np.load(os.path.join(load_path, 'y_train.npy'))
        y_val = np.load(os.path.join(load_path, 'y_val.npy'))
        y_test = np.load(os.path.join(load_path, 'y_test.npy'))
        
        with open(os.path.join(load_path, 'classes.pkl'), 'rb') as f:
            classes = pickle.load(f)
            
        return X_train, X_val, X_test, y_train, y_val, y_test, classes
    
    def visualize_samples(self, X, y, classes, num_samples=16):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(X))):
            axes[i].imshow(X[i])
            axes[i].set_title(f'Class: {classes[np.argmax(y[i])]}')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    # Replace 'your_dataset_path' with the actual path to your Kaggle dataset
    preprocessor = SignLanguagePreprocessor('your_dataset_path')
    
    # Load and preprocess data
    X, y_categorical, y_raw = preprocessor.load_and_preprocess_data(max_samples_per_class=2000)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y_categorical)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Visualize samples
    preprocessor.visualize_samples(X_train, y_train, preprocessor.classes)
    
    print("Preprocessing complete!")