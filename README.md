# Real-Time Sign Language Recognition

![GitHub Repo stars](https://img.shields.io/github/stars/sohail-RM2004/Sign_Language_Detection-using-DL-and-CV?style=social)
![GitHub forks](https://img.shields.io/github/forks/sohail-RM2004/Sign_Language_Detection-using-DL-and-CV?style=social)


---

## Abstract

This project presents a real-time sign language recognition system that converts American Sign Language (ASL) hand gestures into text using deep learning and computer vision techniques. The system employs a Custom CNN to classify 29 hand gestures (A–Z, space, delete, nothing) with high accuracy. Using MediaPipe for hand tracking and OpenCV for video processing, it delivers efficient gesture recognition suitable for accessibility applications. GUI and CLI interfaces, along with text-to-speech functionality, enhance user experience.

---

## ✅ Features

- Real-time gesture recognition via webcam  
- Deep learning model: Custom CNN  
- Supports 29 classes (A-Z, space, delete, nothing)  
- Builds sentences from recognized gestures  
- Text-to-speech output for accessibility  
- Optimized for 25-30 FPS on standard hardware  
- GUI and CLI modes for interaction

---

## 📊 Model Summary

| Model      | Accuracy | Training Time | Parameters |
| ---------- | -------- | ------------- | ---------- |
| Custom CNN | 96.71%  | ~45 min       | 2.1M       |

---

## 🧠 Training Details

**Timestamp:** 2025-09-02_04-41-14  
**Dataset Path:** `datasetasl/asl_alphabet_train/asl_alphabet_train`  
**Image Size:** (64, 64)  
**Batch Size:** 32  
**Epochs:** 14  
**Learning Rate:** 0.001

**Test Accuracy:** 96.71%  
**Test Top-3 Accuracy:** 0.00%  
**Test Loss:** 0.2270

**Main Training Script:** `complete_training_script.py`  
**Webcam Interface Script:** `gui_demo.py`

---

## 📂 Useful Links

<img width="970" height="782" alt="image" src="https://github.com/user-attachments/assets/01d76aba-67b5-478f-9aad-d14387aa57e7" />


- 📄 [Project Report (PDF)](https://drive.google.com/file/d/1BVI9wG_SiH3DaoV2AiGHntW9Wqx7P0QY/view?usp=drive_link)
  
- 🎥 [Demo Video](https://drive.google.com/file/d/1UsTjYHd5Y9sJoXJ4DUOzvhN9LVwvD4a1/view?usp=drive_link)


---

## 🛠 Technologies Used

- Python, TensorFlow, Keras  
- OpenCV, MediaPipe  
- pyttsx3 for text-to-speech  
- Tkinter for GUI

---

## 📄 Conclusion

This project shows the successful development of a real-time ASL recognition system using CNNs and transfer learn-ing. The system reached about 96.7% accuracy, worked at 25–30 FPS, and used efficient hand tracking with a lightweight model, making it suitable for normal computers.
The system is useful as an assistive tool, with both command-line and GUI options, along with text-to-speech for better accessibility. While it currently handles static gestures, it can be extended to support dynamic gestures and multi-hand recognition.


## 📬 Contact

**Mohammed Sohail Rehan**  
m.sohailrehan@gmail.com
