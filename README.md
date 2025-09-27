# Sign Language Recognition System

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure parameters in `config.yaml`:**
   - Adjust epochs, learning rate, image size, etc.
   - Set dataset and model paths

3. **Train model:**
   ```bash
   python complete_training_script.py
   ```

4. **Run real-time detection:**
   ```bash
   python realtime_detection.py
   ```

## Configuration

All parameters are centralized in `config.yaml`:
- **Model**: Architecture, image size, classes
- **Training**: Epochs, batch size, learning rate
- **Detection**: Confidence thresholds, timing
- **Paths**: Dataset, models, output directories

## Core Files

- `config.yaml` - Central configuration
- `config_loader.py` - Configuration utility
- `complete_training_script.py` - Training pipeline
- `sign_language_model.py` - Model architectures
- `sign_language_preprocessing.py` - Data preprocessing
- `realtime_detection.py` - Live detection
- `gui_demo.py` - GUI interface