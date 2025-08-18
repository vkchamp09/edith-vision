EDITH Vision – Iron Man–Style HUD with YOLOv8 & Face Recognition
A cutting-edge Heads-Up Display (HUD) system inspired by EDITH from Spider-Man: Far From Home, integrating real-time object detection, face recognition, text detection, and voice-controlled AI to create an interactive augmented reality experience.
🚀 Features
Face Recognition – Identify and differentiate between known and unknown faces.
Object Detection – Real-time detection of multiple objects using YOLOv8.
Text Detection – Capture and process text from images or video feeds.
Voice Interaction – Control the system and receive feedback via voice commands.
Chatbot Integration – Converse with an AI-powered chatbot for intelligent responses.

edith-vision/
│
├── assets/                 # Images, icons, and visual assets
├── faces/                  # Known and unknown faces for recognition
├── text_detection/         # Modules for OCR and text processing
├── edith.py                # Main application script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables configuration
├── README.md               # Project documentation
└── session_logs/           # Logs of user sessions

💻 Tech Stack
Language: Python 3.x
Computer Vision: OpenCV, YOLOv8
Face Recognition: face_recognition library
OCR/Text Detection: pytesseract
Voice Interaction: pyttsx3, SpeechRecognition
AI Chatbot: ChatterBot

⚡ Installation & Setup (Anaconda)
Clone the Repository
git clone https://github.com/vkchamp09/edith-vision.git
cd edith-vision
Create a Conda Environment
conda create -n edith-vision python=3.11
Activate the Environment
conda activate edith-vision

Configure Environment Variables
Rename .env.example to .env
Add your API keys or configuration values (if any)
Run the Application
python edith.py

🛠 Usage
The system will start the HUD interface with live camera feed.
Voice Commands: Speak commands to interact with the AI assistant.
Face Detection: Automatically identifies known faces and logs unknown faces.
Object Detection: Recognizes and highlights objects in real-time.
Text Detection: Detects text from images and displays results.

<img width="955" height="566" alt="Screenshot 2025-08-18 at 6 10 08 PM" src="https://github.com/user-attachments/assets/f002a367-72df-442d-bf90-2da7d9c5c30f" />

🤝 Contributing
Contributions are welcome! Here’s how you can help:
Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Create a Pull Request.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

💡 Future Enhancements
Integration with AR glasses for wearable HUD experience.
Gesture controls for hands-free interaction.
Enhanced multi-language voice recognition.
Improved object detection accuracy with custom YOLOv8 models.
📌 References
YOLOv8 Official Docs
OpenCV Python Docs
ChatterBot
Inspired by EDITH in Spider-Man: Far From Home
