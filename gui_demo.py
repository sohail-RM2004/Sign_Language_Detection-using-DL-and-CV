import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
from PIL import Image, ImageTk
import pickle
import pyttsx3
import os

class SignLanguageGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)  # Reduced minimum size for better compatibility
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.model = None
        self.classes = None
        self.video_capture = None
        self.is_detecting = False
        self.current_frame = None
        self.current_photo = None
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_available = True
        except:
            self.tts_available = False
        
        # Word formation
        self.current_word = tk.StringVar(value="")
        self.sentence = tk.StringVar(value="")
        self.prediction = tk.StringVar(value="No prediction")
        self.confidence = tk.StringVar(value="0.00")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface with proper responsive design"""
        # Configure root grid weights for proper resizing
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
        
        title_label = tk.Label(
            title_frame, 
            text="Sign Language Recognition System", 
            font=('Arial', 18, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack()
        
        # Main container using grid for better control
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        # Configure main frame grid
        main_frame.grid_columnconfigure(0, weight=2)  # Left panel takes more space
        main_frame.grid_columnconfigure(1, weight=1)  # Right panel fixed width
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Video and controls
        left_panel = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.grid_rowconfigure(1, weight=1)  # Video frame expands
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Video label
        video_label = tk.Label(left_panel, text="Camera Feed", font=('Arial', 12, 'bold'), 
                              fg='#ecf0f1', bg='#34495e')
        video_label.grid(row=0, column=0, pady=(10, 5))
        
        # Video frame container
        video_container = tk.Frame(left_panel, bg='#34495e')
        video_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)
        
        # Video frame with responsive sizing
        self.video_frame = tk.Label(video_container, bg='black', relief=tk.SUNKEN, bd=3,
                                   text="Camera Feed\nLoad model and start detection")
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        
        # Control buttons frame
        control_frame = tk.Frame(left_panel, bg='#34495e')
        control_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(5, 15))
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
        
        # Buttons with consistent sizing
        button_style = {
            'font': ('Arial', 10, 'bold'),
            'relief': tk.FLAT,
            'pady': 8,
            'width': 12  # Fixed width for consistency
        }
        
        self.start_btn = tk.Button(
            control_frame, 
            text="▶ Start", 
            command=self.start_detection,
            bg='#27ae60', fg='white',
            **button_style
        )
        self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.stop_btn = tk.Button(
            control_frame, 
            text="⏹ Stop", 
            command=self.stop_detection,
            bg='#e74c3c', fg='white',
            state=tk.DISABLED,
            **button_style
        )
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        load_model_btn = tk.Button(
            control_frame, 
            text=" Load Model", 
            command=self.load_model,
            bg='#3498db', fg='white',
            **button_style
        )
        load_model_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Right panel - Information and results with fixed width
        right_panel = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_panel.grid_rowconfigure(6, weight=1)  # Instructions section expands
        right_panel.configure(width=320)  # Fixed width
        right_panel.grid_propagate(False)  # Maintain fixed width
        
        # Model info section
        self.create_info_section(right_panel)
        
        # Prediction results section
        self.create_prediction_section(right_panel)
        
        # Text formation section
        self.create_text_section(right_panel)
        
        # Instructions section
        self.create_instructions_section(right_panel)
    
    def create_info_section(self, parent):
        """Create model information section"""
        info_label = tk.Label(parent, text="Model Information", font=('Arial', 11, 'bold'), 
                             fg='#ecf0f1', bg='#34495e')
        info_label.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        
        self.model_status = tk.Label(parent, text="No model loaded", 
                                    fg='#e74c3c', bg='#34495e', font=('Arial', 9),
                                    wraplength=280)
        self.model_status.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
        
        separator1 = ttk.Separator(parent, orient='horizontal')
        separator1.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
    
    def create_prediction_section(self, parent):
        """Create prediction results section"""
        results_label = tk.Label(parent, text="Live Prediction", font=('Arial', 11, 'bold'), 
                                fg='#ecf0f1', bg='#34495e')
        results_label.grid(row=3, column=0, pady=(5, 5), sticky="ew")
        
        # Prediction container
        pred_container = tk.Frame(parent, bg='#34495e')
        pred_container.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        pred_container.grid_columnconfigure(0, weight=1)
        
        # Current prediction
        tk.Label(pred_container, text="Prediction:", font=('Arial', 9, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=0, column=0, sticky="w")
        
        pred_value = tk.Label(pred_container, textvariable=self.prediction, 
                             font=('Arial', 12, 'bold'), fg='#f39c12', bg='#34495e',
                             wraplength=260)
        pred_value.grid(row=1, column=0, sticky="w")
        
        # Confidence
        tk.Label(pred_container, text="Confidence:", font=('Arial', 9, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=2, column=0, sticky="w", pady=(5, 0))
        
        conf_value = tk.Label(pred_container, textvariable=self.confidence, 
                             font=('Arial', 11), fg='#2ecc71', bg='#34495e')
        conf_value.grid(row=3, column=0, sticky="w")
        
        separator2 = ttk.Separator(parent, orient='horizontal')
        separator2.grid(row=5, column=0, sticky="ew", padx=10, pady=8)
    
    def create_text_section(self, parent):
        """Create text formation section"""
        text_label = tk.Label(parent, text="Text Formation", font=('Arial', 11, 'bold'), 
                             fg='#ecf0f1', bg='#34495e')
        text_label.grid(row=6, column=0, pady=(5, 5), sticky="ew")
        
        text_container = tk.Frame(parent, bg='#34495e')
        text_container.grid(row=7, column=0, sticky="ew", padx=10, pady=5)
        text_container.grid_columnconfigure(0, weight=1)
        
        # Current word
        tk.Label(text_container, text="Current Word:", font=('Arial', 9, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=0, column=0, sticky="w")
        
        word_entry = tk.Entry(text_container, textvariable=self.current_word, 
                             font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1', 
                             relief=tk.FLAT, bd=5)
        word_entry.grid(row=1, column=0, sticky="ew", pady=2)
        
        # Sentence
        tk.Label(text_container, text="Sentence:", font=('Arial', 9, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=2, column=0, sticky="w", pady=(8, 0))
        
        sentence_text = tk.Text(text_container, height=3, font=('Arial', 9), 
                               bg='#2c3e50', fg='#ecf0f1', relief=tk.FLAT, bd=5,
                               wrap=tk.WORD)
        sentence_text.grid(row=3, column=0, sticky="ew", pady=2)
        self.sentence_text = sentence_text
        
        # Text control buttons
        button_container = tk.Frame(text_container, bg='#34495e')
        button_container.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        button_container.grid_columnconfigure(0, weight=1)
        button_container.grid_columnconfigure(1, weight=1)
        if self.tts_available:
            button_container.grid_columnconfigure(2, weight=1)
        
        clear_word_btn = tk.Button(
            button_container, 
            text="Clear Word", 
            command=self.clear_word,
            bg='#f39c12', fg='white', font=('Arial', 8),
            relief=tk.FLAT, width=8
        )
        clear_word_btn.grid(row=0, column=0, padx=1, sticky="ew")
        
        clear_sentence_btn = tk.Button(
            button_container, 
            text="Clear All", 
            command=self.clear_sentence,
            bg='#e67e22', fg='white', font=('Arial', 8),
            relief=tk.FLAT, width=8
        )
        clear_sentence_btn.grid(row=0, column=1, padx=1, sticky="ew")
        
        if self.tts_available:
            speak_btn = tk.Button(
                button_container, 
                text="🔊 Speak", 
                command=self.speak_sentence,
                bg='#9b59b6', fg='white', font=('Arial', 8),
                relief=tk.FLAT, width=8
            )
            speak_btn.grid(row=0, column=2, padx=1, sticky="ew")
    
    def create_instructions_section(self, parent):
        """Create instructions section"""
        separator3 = ttk.Separator(parent, orient='horizontal')
        separator3.grid(row=8, column=0, sticky="ew", padx=10, pady=8)
        
        instructions_frame = tk.Frame(parent, bg='#34495e')
        instructions_frame.grid(row=9, column=0, sticky="nsew", padx=10, pady=(5, 10))
        instructions_frame.grid_columnconfigure(0, weight=1)
        instructions_frame.grid_rowconfigure(1, weight=1)
        
        tk.Label(instructions_frame, text="Instructions:", font=('Arial', 10, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=0, column=0, sticky="w")
        
        # Scrollable instructions
        inst_frame = tk.Frame(instructions_frame, bg='#34495e')
        inst_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        
        instructions = [
            "1. Load your trained model",
            "2. Start detection",
            "3. Show hand gestures to camera",
            "4. 'space' = add space",
            "5. 'del' = delete character",
            "",
            "Note: High confidence (>0.8)",
            "predictions are processed"
        ]
        
        for i, instruction in enumerate(instructions):
            tk.Label(inst_frame, text=instruction, font=('Arial', 8),
                    fg='#bdc3c7', bg='#34495e', wraplength=280,
                    justify='left').grid(row=i, column=0, sticky="w", pady=1)
    
    def load_model(self):
        """Load trained model and classes"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras files", "*.keras"), ("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                self.model = load_model(model_path)
                
                # Try to load classes file from same directory
                if model_path.endswith('.keras'):
                    classes_path = model_path.replace('.keras', '_classes.pkl')
                else:
                    classes_path = model_path.replace('.h5', '_classes.pkl')
                
                models_dir = os.path.dirname(model_path)
                general_classes_path = os.path.join(models_dir, "classes.pkl")
                
                # Check if classes file exists
                if os.path.exists(classes_path):
                    if not tk.messagebox.askyesno("Classes File", 
                        f"Load classes from:\n{os.path.basename(classes_path)}"):
                        classes_path = filedialog.askopenfilename(
                            title="Select Classes File",
                            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                        )
                elif os.path.exists(general_classes_path):
                    if tk.messagebox.askyesno("Classes File", 
                        f"Found general classes file. Load it?"):
                        classes_path = general_classes_path
                    else:
                        classes_path = filedialog.askopenfilename(
                            title="Select Classes File",
                            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                        )
                else:
                    classes_path = filedialog.askopenfilename(
                        title="Select Classes File",
                        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                    )
                
                if classes_path:
                    with open(classes_path, 'rb') as f:
                        self.classes = pickle.load(f)
                else:
                    # Default classes
                    self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                                  'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
                
                self.model_status.config(text="Model loaded successfully", fg='#2ecc71')
                messagebox.showinfo("Success", "Model loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.model_status.config(text="Failed to load model", fg='#e74c3c')
    
    def start_detection(self):
        """Start real-time detection"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise Exception("Could not open camera")
                
            self.is_detecting = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_detection(self):
        """Stop real-time detection"""
        self.is_detecting = False
        if self.video_capture:
            self.video_capture.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_frame.config(image='', text="Camera Feed\nLoad model and start detection")
        self.current_photo = None
    
    def detection_loop(self):
        """Main detection loop with proper error handling"""
        try:
            while self.is_detecting and self.video_capture and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                                  self.mp_hands.HAND_CONNECTIONS)
                        
                        # Extract and predict
                        roi = self.extract_hand_roi(frame, hand_landmarks)
                        if roi is not None:
                            predicted_class, conf = self.predict_sign(roi)
                            
                            # Update UI in main thread
                            self.root.after(0, self.update_ui, predicted_class, conf)
                            
                            # Update text formation
                            if conf > 0.95:  # High confidence threshold
                                self.root.after(0, self.update_text, predicted_class)
                
                # Get video frame dimensions dynamically
                frame_widget_width = self.video_frame.winfo_width()
                frame_widget_height = self.video_frame.winfo_height()
                
                # Only resize if we have valid dimensions
                if frame_widget_width > 1 and frame_widget_height > 1:
                    # Convert and resize frame for display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Calculate aspect ratio preserving resize
                    h, w = frame.shape[:2]
                    aspect_ratio = w / h
                    
                    if frame_widget_width / frame_widget_height > aspect_ratio:
                        # Height is the limiting factor
                        new_height = frame_widget_height - 10  # Small margin
                        new_width = int(new_height * aspect_ratio)
                    else:
                        # Width is the limiting factor
                        new_width = frame_widget_width - 10  # Small margin
                        new_height = int(new_width / aspect_ratio)
                    
                    frame = cv2.resize(frame, (new_width, new_height))
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update video frame in main thread
                    self.root.after(0, self.update_video_frame, photo)
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Error in detection loop: {e}")
            self.root.after(0, self.stop_detection)
    
    def update_ui(self, predicted_class, confidence):
        """Update UI elements in main thread"""
        try:
            self.prediction.set(predicted_class)
            self.confidence.set(f"{confidence:.2f}")
        except Exception as e:
            print(f"Error updating UI: {e}")
    
    def update_video_frame(self, photo):
        """Update video frame in main thread"""
        try:
            self.current_photo = photo
            self.video_frame.config(image=photo, text="")
        except Exception as e:
            print(f"Error updating video frame: {e}")
    
    def extract_hand_roi(self, frame, hand_landmarks):
        """Extract hand region of interest"""
        h, w, c = frame.shape
        
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        padding = 0.1
        x_min = max(0, int((min(x_coords) - padding) * w))
        x_max = min(w, int((max(x_coords) + padding) * w))
        y_min = max(0, int((min(y_coords) - padding) * h))
        y_max = min(h, int((max(y_coords) + padding) * h))
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype('float32') / 255.0
            return roi
        
        return None
    
    def predict_sign(self, roi):
        """Predict sign from ROI"""
        roi_batch = np.expand_dims(roi, axis=0)
        prediction = self.model.predict(roi_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        return self.classes[predicted_class_idx], confidence
    
    def update_text(self, predicted_class):
        """Update current word and sentence"""
        if predicted_class == "space":
            current = self.current_word.get()
            if current:
                self.sentence_text.insert(tk.END, current + " ")
                self.sentence_text.see(tk.END)
                self.current_word.set("")
        elif predicted_class == "del":
            current = self.current_word.get()
            if current:
                self.current_word.set(current[:-1])
            else:
                content = self.sentence_text.get("1.0", tk.END).strip()
                if content:
                    words = content.split()
                    if words:
                        self.sentence_text.delete("1.0", tk.END)
                        self.sentence_text.insert("1.0", " ".join(words[:-1]) + " ")
        elif predicted_class != "nothing":
            current = self.current_word.get()
            self.current_word.set(current + predicted_class.upper())
    
    def clear_word(self):
        """Clear current word"""
        self.current_word.set("")
    
    def clear_sentence(self):
        """Clear sentence and current word"""
        self.current_word.set("")
        self.sentence_text.delete("1.0", tk.END)
    
    def speak_sentence(self):
        """Speak the current sentence"""
        if not self.tts_available:
            messagebox.showwarning("Warning", "Text-to-speech not available!")
            return
        
        def speak():
            try:
                sentence = self.sentence_text.get("1.0", tk.END).strip()
                current_word = self.current_word.get()
                full_text = sentence + " " + current_word if sentence else current_word
                
                if full_text.strip():
                    self.tts_engine.say(full_text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_detection()
        if hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.stop()
            except:
                pass
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    try:
        app = SignLanguageGUI()
        app.run()
    except Exception as e:
        print(f"Error running GUI: {e}")
        import traceback
        traceback.print_exc()



