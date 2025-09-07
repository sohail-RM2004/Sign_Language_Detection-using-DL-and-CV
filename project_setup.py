#!/usr/bin/env python3
"""
Sign Language Recognition Project Setup Script
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

class ProjectSetup:
    def __init__(self):
        self.project_name = "SignLanguageRecognition"
        self.project_dir = Path(self.project_name)
        
    def create_project_structure(self):
        """Create the project directory structure"""
        print("🏗️  Creating project structure...")
        
        # Create main directories
        directories = [
            self.project_dir,
            self.project_dir / "models",
            self.project_dir / "data",
            self.project_dir / "plots",
            self.project_dir / "scripts",
            self.project_dir / "demo_videos",
            self.project_dir / "documentation"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print("\n Creating requirements.txt...")
        
        requirements = [
            "tensorflow>=2.8.0",
            "opencv-python>=4.5.0",
            "mediapipe>=0.8.0",
            "numpy>=1.21.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "pillow>=8.0.0",
            "pyttsx3>=2.90",
            "tk",  # Usually comes with Python
        ]
        
        req_file = self.project_dir / "requirements.txt"
        with open(req_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements))
        
        print(f"   Created: {req_file}")
    
    def create_config_file(self):
        """Create configuration file"""
        print("\n  Creating config.py...")
        
        config_content = '''"""
Configuration file for Sign Language Recognition Project

"""

# Dataset Configuration
DATASET_PATH = "path"
MAX_SAMPLES_PER_CLASS = 1500  # Reduce for faster training on limited resources
IMAGE_SIZE = (64, 64)  # Input image size for the model

# Training Configuration
BATCH_SIZE = 32  # Reduce if you have memory issues
EPOCHS = 25  # Number of training epochs
LEARNING_RATE = 0.001
TEST_SIZE = 0.2  # Percentage of data for testing
VALIDATION_SIZE = 0.1  # Percentage of data for validation

# Model Configuration
NUM_CLASSES = 29  # A-Z + space + del + nothing
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for predictions

# Real-time Detection Configuration
DETECTION_CONFIDENCE = 0.7  # MediaPipe hand detection confidence
TRACKING_CONFIDENCE = 0.5   # MediaPipe hand tracking confidence
PREDICTION_DELAY = 1.5      # Seconds between word updates
SMOOTH_PREDICTIONS = True   # Enable prediction smoothing

# Paths
MODEL_SAVE_PATH = "models/best_sign_model.h5"
CLASSES_SAVE_PATH = "data/classes.pkl"
LOG_DIR = "logs"

# Performance Settings
USE_GPU = True  # Set to False if you don't have a GPU
MIXED_PRECISION = False  # Enable for faster training (if supported)

# Demo Settings
DEMO_VIDEO_PATH = "demo_videos"
ENABLE_TTS = True  # Text-to-speech functionality
'''
        
        config_file = self.project_dir / "config.py"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"   Created: {config_file}")
    
    def create_readme(self):
        """Create comprehensive README.md"""
        print("\n Creating README.md...")
        
        readme_content = '''# 🤟 Sign Language Recognition System

A real-time sign language recognition system using deep learning, computer vision, and MediaPipe for hand tracking.

##  Features

- **Real-time Detection**: Live hand gesture recognition through webcam
- **Deep Learning**: CNN and Transfer Learning models for accurate classification
- **29 Classes**: A-Z letters + space + delete + nothing gestures
- **Text Formation**: Automatic word and sentence formation
- **Text-to-Speech**: Voice output for recognized text
- **User-friendly GUI**: Easy-to-use graphical interface
- **Performance Optimization**: Designed for limited computational resources

##  Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd SignLanguageRecognition

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
- Download the ASL dataset from Kaggle
- Update the `DATASET_PATH` in `config.py`
- Ensure dataset structure: `dataset/A/`, `dataset/B/`, etc.

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run Real-time Detection
```bash
# Command line version
python real_time_detection.py

# GUI version  
python gui_demo.py
```

##  Project Structure

```
SignLanguageRecognition/
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── train_model.py              # Complete training script
├── real_time_detection.py      # Command-line detection
├── gui_demo.py                 # GUI application
├── models/                     # Saved trained models
├── data/                       # Preprocessed data and classes
├── plots/                      # Training plots and analysis
├── demo_videos/               # Demo recordings
└── documentation/             # Additional documentation
```

##  Usage Instructions

### Training
1. Update dataset path in `config.py`
2. Run `python train_model.py`
3. The script will train both CNN and transfer learning models
4. Best model will be saved automatically

### Real-time Detection
- **GUI Version**: Run `python gui_demo.py` for user-friendly interface
- **Command Line**: Run `python real_time_detection.py` for terminal-based detection

### Controls
- **Space gesture**: Add space between words
- **Del gesture**: Delete last character/word
- **Clear buttons**: Reset current word or entire sentence
- **Speak button**: Text-to-speech output

##  Model Architecture

### Custom CNN
- 3 Convolutional blocks with BatchNormalization
- Global Average Pooling for efficiency
- Dense layers with dropout for regularization
- Optimized for limited resources

### Transfer Learning
- MobileNetV2 base model (pre-trained on ImageNet)
- Custom classifier head
- Faster training with fewer epochs

##  Performance Tips

### For Limited Resources:
- Reduce `MAX_SAMPLES_PER_CLASS` in config
- Use smaller `BATCH_SIZE` (16 or 8)
- Enable mixed precision training
- Use transfer learning model

### For Better Accuracy:
- Increase dataset size
- Use data augmentation
- Fine-tune hyperparameters
- Collect more diverse training data

##  Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce batch size and max samples
2. **Camera not detected**: Check camera permissions and connections
3. **Model loading fails**: Verify file paths and model compatibility
4. **Low accuracy**: Ensure good lighting and clear hand gestures

### Performance Optimization:
- Use GPU if available (set `USE_GPU = True`)
- Close other applications during training
- Ensure good lighting for real-time detection
- Keep hand gestures clear and stable

##  Demo

Create demo videos showing:
1. Training process and results
2. Real-time detection in action  
3. Different lighting conditions
4. Word and sentence formation
5. Text-to-speech functionality

##  Results Analysis

The system provides:
- Training accuracy and loss plots
- Confusion matrix analysis
- Classification reports
- Model comparison metrics
- Performance benchmarks

##  Future Enhancements

- [ ] Support for dynamic gestures
- [ ] Multi-hand detection
- [ ] Gesture sequence recognition
- [ ] Mobile app development
- [ ] Cloud deployment
- [ ] Real-time translation to multiple languages

##  Contributing

Feel free to contribute by:
- Adding new gesture classes
- Improving model architecture
- Enhancing UI/UX
- Adding new features
- Optimizing performance

##  License

This project is for educational purposes. Please respect the original dataset licenses.

---

'''
        
        readme_file = self.project_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"   Created: {readme_file}")
    
    def create_main_script(self):
        """Create main execution script"""
        print("\n Creating main.py...")
        
        main_content = '''#!/usr/bin/env python3
"""
Main execution script for Sign Language Recognition Project
This script provides a menu-driven interface to run different parts of the project.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║              🤟 SIGN LANGUAGE RECOGNITION 🤟            ║
    ║                                                          ║
    ║          Real-time Hand Gesture Recognition System       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow
        import cv2
        import mediapipe
        import numpy
        import matplotlib
        import sklearn
        print(" All dependencies are installed!")
        return True
    except ImportError as e:
        print(f" Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main_menu():
    """Display main menu and handle user selection"""
    while True:
        print("\\n" + "="*60)
        print("               MAIN MENU")
        print("="*60)
        print("1.   Train Models")
        print("2.   Real-time Detection (Command Line)")
        print("3.   Launch GUI Application")
        print("4.   View Training Results")
        print("5.   Setup/Configuration")
        print("6.   View Documentation")
        print("7.   Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                print("\\n  Starting model training...")
                subprocess.run([sys.executable, "train_model.py"])
                
            elif choice == '2':
                print("\\n  Starting real-time detection...")
                subprocess.run([sys.executable, "real_time_detection.py"])
                
            elif choice == '3':
                print("\\n  Launching GUI application...")
                subprocess.run([sys.executable, "gui_demo.py"])
                
            elif choice == '4':
                print("\\n  Opening training results...")
                plots_dir = Path("plots")
                if plots_dir.exists():
                    print(f"Training plots available in: {plots_dir}")
                    # Open plots directory
                    if os.name == 'nt':  # Windows
                        os.startfile(plots_dir)
                    elif os.name == 'posix':  # macOS and Linux
                        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', plots_dir])
                else:
                    print("No training results found. Please train a model first.")
                
            elif choice == '5':
                print("\\n  Configuration options:")
                print("- Edit config.py to modify training parameters")
                print("- Update dataset path in config.py")
                print("- Adjust model settings for your hardware")
                input("Press Enter to continue...")
                
            elif choice == '6':
                print("\\n  Documentation:")
                print("- README.md: Complete project documentation")
                print("- config.py: Configuration parameters")
                print("- Check plots/ for training analysis")
                input("Press Enter to continue...")
                
            elif choice == '7':
                print("\\n Thank you for using Sign Language Recognition System!")
                break
                
            else:
                print(" Invalid choice. Please enter a number between 1-7.")
                
        except KeyboardInterrupt:
            print("\\n\\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    print_banner()
    
    if not check_dependencies():
        sys.exit(1)
    
    print(" Welcome to Sign Language Recognition System!")
    print("This system helps convert hand gestures to text in real-time.")
    
    main_menu()
'''
        
        main_file = self.project_dir / "main.py"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_content)
        
        print(f"   Created: {main_file}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("\\n Installing dependencies...")
        try:
            req_file = self.project_dir / "requirements.txt"
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            print("   Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"   Failed to install dependencies: {e}")
    
    def setup_complete_project(self, install_deps=False):
        """Setup the complete project"""
        print(" Setting up Sign Language Recognition Project...")
        
        self.create_project_structure()
        self.create_requirements_file()
        self.create_config_file()
        self.create_readme()
        self.create_main_script()
        
        if install_deps:
            self.install_dependencies()
        
        print("\\n" + "="*60)
        print(" PROJECT SETUP COMPLETE!")
        print("="*60)
        print(f" Project created in: {self.project_dir.absolute()}")
        print("\\n Next steps:")
        print("1. Update dataset path in config.py")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run python main.py to start")
        print("\\n For impressive results:")
        print("- Use good lighting for training and testing")
        print("- Ensure clear hand gestures")
        print("- Train with diverse backgrounds")
        print("- Test in different environments")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Sign Language Recognition Project Setup")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install dependencies after setup")
    args = parser.parse_args()
    
    setup = ProjectSetup()
    setup.setup_complete_project(install_deps=args.install_deps)

if __name__ == "__main__":
    main()