import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import pyttsx3
from collections import deque
import threading
import pickle

class RealTimeSignDetection:
    def __init__(self, model_path, classes_path=None, img_size=(64, 64)):
        self.model = load_model(model_path)
        self.img_size = img_size
        
        # Load class names
        if classes_path:
            with open(classes_path, 'rb') as f:
                self.classes = pickle.load(f)
        else:
            self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                           'space', 'del', 'nothing']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Word formation
        self.current_word = ""
        self.sentence = ""
        self.prediction_history = deque(maxlen=10)  # Smooth predictions
        self.last_prediction_time = time.time()
        self.prediction_confidence_threshold = 0.8
        self.stable_prediction_count = 0
        self.required_stable_predictions = 5
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        
    def extract_hand_roi(self, frame, hand_landmarks):
        """Extract hand region of interest from frame"""
        h, w, c = frame.shape
        
        # Get hand bounding box
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        # Add padding
        padding = 0.1
        x_min = max(0, int((min(x_coords) - padding) * w))
        x_max = min(w, int((max(x_coords) + padding) * w))
        y_min = max(0, int((min(y_coords) - padding) * h))
        y_max = min(h, int((max(y_coords) + padding) * h))
        
        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            # Resize to model input size
            roi = cv2.resize(roi, self.img_size)
            roi = roi.astype('float32') / 255.0
            return roi, (x_min, y_min, x_max, y_max)
        
        return None, None
    
    def predict_sign(self, roi):
        """Predict sign from hand ROI"""
        if roi is None:
            return None, 0.0
            
        # Prepare image for prediction
        roi_batch = np.expand_dims(roi, axis=0)
        
        # Make prediction
        prediction = self.model.predict(roi_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        return self.classes[predicted_class_idx], confidence
    
    def smooth_predictions(self, predicted_class, confidence):
        """Smooth predictions to reduce noise"""
        if confidence > self.prediction_confidence_threshold:
            self.prediction_history.append(predicted_class)
            
            # Get most common prediction
            if len(self.prediction_history) >= 3:
                most_common = max(set(self.prediction_history), 
                                key=self.prediction_history.count)
                return most_common
        
        return predicted_class if confidence > 0.5 else "nothing"
    
    def update_word(self, predicted_class):
        """Update current word based on prediction"""
        current_time = time.time()
        
        # Check if enough time has passed since last prediction
        if current_time - self.last_prediction_time > 1.5:  # 1.5 second delay
            if predicted_class == "space":
                if self.current_word:
                    self.sentence += self.current_word + " "
                    self.current_word = ""
            elif predicted_class == "del":
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                elif self.sentence:
                    words = self.sentence.strip().split()
                    if words:
                        self.sentence = " ".join(words[:-1]) + " "
            elif predicted_class != "nothing":
                self.current_word += predicted_class.upper()
                
            self.last_prediction_time = current_time
    
    def speak_text(self, text):
        """Speak text using text-to-speech (runs in separate thread)"""
        def speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def draw_info(self, frame, predicted_class, confidence, bbox=None):
        """Draw prediction info on frame"""
        h, w, c = frame.shape
        
        # Draw hand bounding box
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Create info panel
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Prediction info
        pred_text = f"Prediction: {predicted_class} ({confidence:.2f})"
        cv2.putText(panel, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current word
        word_text = f"Current Word: {self.current_word}"
        cv2.putText(panel, word_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Sentence
        sentence_text = f"Sentence: {self.sentence}"
        if len(sentence_text) > 50:
            sentence_text = sentence_text[:47] + "..."
        cv2.putText(panel, sentence_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # FPS
        if self.fps_counter:
            fps = len(self.fps_counter) / sum(self.fps_counter)
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(panel, fps_text, (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine frame and panel
        combined_frame = np.vstack([frame, panel])
        return combined_frame
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting real-time sign language detection...")
        print("Controls:")
        print("- 'q': Quit")
        print("- 's': Speak current sentence")
        print("- 'c': Clear sentence")
        print("- 'r': Reset current word")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            predicted_class = "nothing"
            confidence = 0.0
            bbox = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                              self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract hand ROI and predict
                    roi, bbox = self.extract_hand_roi(frame, hand_landmarks)
                    if roi is not None:
                        predicted_class, confidence = self.predict_sign(roi)
                        predicted_class = self.smooth_predictions(predicted_class, confidence)
                        
                        # Update word
                        if confidence > self.prediction_confidence_threshold:
                            self.update_word(predicted_class)
            
            # Draw information
            display_frame = self.draw_info(frame, predicted_class, confidence, bbox)
            
            # Show frame
            cv2.imshow('Sign Language Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                full_text = self.sentence + self.current_word
                if full_text.strip():
                    print(f"Speaking: {full_text}")
                    self.speak_text(full_text)
            elif key == ord('c'):
                self.sentence = ""
                self.current_word = ""
                print("Sentence cleared")
            elif key == ord('r'):
                self.current_word = ""
                print("Current word reset")
            
            # Calculate FPS
            frame_time = time.time() - start_time
            self.fps_counter.append(frame_time)
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = RealTimeSignDetection(
        model_path="best_sign_model.keras",  # Path to your trained model
        classes_path="preprocessed_data/classes.pkl"  # Path to classes file
    )
    
    # Run detection
    detector.run_detection()