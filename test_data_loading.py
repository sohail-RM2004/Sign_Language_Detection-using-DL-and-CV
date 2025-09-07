#!/usr/bin/env python3
"""
Test script to verify data loading works correctly
"""

import os
import sys
from sign_language_preprocessing import SignLanguagePreprocessor

def test_data_loading():
    """Test if data can be loaded correctly"""
    print(" Testing Data Loading...")
    
    # Test dataset path
    dataset_path = "datasetasl/asl_alphabet_train/asl_alphabet_train"
    
    if not os.path.exists(dataset_path):
        print(f" Dataset path not found: {dataset_path}")
        return False
    
    print(f" Dataset path exists: {dataset_path}")
    
    # Check if subdirectories exist
    expected_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                       'space', 'del', 'nothing']
    
    missing_classes = []
    for class_name in expected_classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
        else:
            # Count images in this class
            try:
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   {class_name}: {len(image_files)} images")
            except Exception as e:
                print(f"   {class_name}: Error reading directory - {e}")
                missing_classes.append(class_name)
    
    if missing_classes:
        print(f" Missing classes: {missing_classes}")
        return False
    
    print(" All expected classes found!")
    
    # Test preprocessing
    try:
        print("\n Testing preprocessing...")
        preprocessor = SignLanguagePreprocessor(dataset_path)
        
        # Load a small sample for testing
        X, y_categorical, y_raw = preprocessor.load_and_preprocess_data(max_samples_per_class=100)
        
        print(f" Data loaded successfully!")
        print(f"   X shape: {X.shape}")
        print(f"   y_categorical shape: {y_categorical.shape}")
        print(f"   y_raw shape: {y_raw.shape}")
        print(f"   Number of classes: {len(preprocessor.classes)}")
        
        # Test data splitting
        print("\n Testing data splitting...")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y_categorical, test_size=0.2, val_size=0.1
        )
        
        print(f" Data split successfully!")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        return True
        
    except Exception as e:
        print(f" Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    
    if success:
        print("\n All tests passed! Data loading is working correctly.")
        print("You can now run the complete training script.")
    else:
        print("\n Tests failed! Please fix the issues before running training.")
        sys.exit(1) 