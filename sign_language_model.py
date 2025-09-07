import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                   Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping, 
                                       ModelCheckpoint)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np

class SignLanguageModel:
    def __init__(self, num_classes=29, img_size=(64, 64)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def create_efficient_cnn(self):
        """
        Create an efficient CNN model optimized for limited resources
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            # Classifier
            GlobalAveragePooling2D(),  # More efficient than Flatten + Dense
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def create_transfer_learning_model(self):
        """
        Create a model using MobileNetV2 for better performance with less training time
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(input_shape=(*self.img_size, 3),
                               include_top=False,
                               weights='imagenet')
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classifier
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and metrics"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_efficient_cnn() or create_transfer_learning_model() first.")
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model compiled with learning rate: {learning_rate}")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def get_callbacks(self, model_path="custom_cnn_model.keras"):
        """Get training callbacks for optimization"""
        callbacks = [
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """Train the model with given data"""
        if self.model is None:
            raise ValueError("Model not created yet.")
            
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip for sign language
            fill_mode='nearest'
        )
        
        # Fit the data generator
        datagen.fit(X_train)
        
        print(f"Starting training with batch size: {batch_size}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy, 0.0  # Return 0.0 for top3_accuracy to maintain compatibility
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-3 Accuracy (if available)
        if 'top_3_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            axes[1, 0].set_title('Model Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Top-3 Accuracy\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Model Top-3 Accuracy')
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image, classes):
        """Predict single image"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Ensure image is in correct format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        return classes[predicted_class], confidence, prediction[0]

# Example usage
if __name__ == "__main__":
    # Initialize model
    sign_model = SignLanguageModel(num_classes=29)
    
    # Create model (choose one)
    model = sign_model.create_efficient_cnn()  # For custom CNN
    # model = sign_model.create_transfer_learning_model()  # For transfer learning
    
    # Compile model
    sign_model.compile_model(learning_rate=0.001)
    
    # Print model summary
    model.summary()
    
