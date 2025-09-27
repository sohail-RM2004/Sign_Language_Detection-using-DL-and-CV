import yaml
import os

class Config:
    def __init__(self, config_path="config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config if file doesn't exist
            self.config = self._default_config()
    
    def _default_config(self):
        return {
            'model': {'img_size': [128, 128], 'num_classes': 29, 'architecture': 'efficient_cnn'},
            'training': {'epochs': 25, 'batch_size': 16, 'learning_rate': 0.0005, 'max_samples_per_class': 2000, 'validation_split': 0.1, 'test_split': 0.2},
            'detection': {'confidence_threshold': 0.4, 'prediction_threshold': 0.3, 'prediction_history_size': 5, 'update_delay': 1.0},
            'paths': {'dataset': 'datasetasl/asl_alphabet_train/asl_alphabet_train', 'save_dir': 'sign_language_project', 'model_file': 'best_sign_model.keras', 'classes_file': 'sign_language_project/classes.pkl'},
            'mediapipe': {'max_num_hands': 1, 'min_detection_confidence': 0.7, 'min_tracking_confidence': 0.5}
        }
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global config instance
config = Config()