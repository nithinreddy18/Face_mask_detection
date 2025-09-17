# 🛡️ Face Mask Detection

This project detects whether a person is wearing a **face mask** or not using **OpenCV** and **deep learning**.  
It uses a pre-trained model for face mask classification and OpenCV's Haar Cascade for face detection.  

---

## 📂 Project Structure

face_mask_detection/
│── main.py # Main script to run face mask detection
│── train_model.py # (Optional) Script to train model from scratch
│── requirements.txt # Python dependencies
│── models/
│ └── mask_detector.model # Trained mask detection model
│── haarcascades/
│ └── haarcascade_frontalface_default.xml # Face detection cascade
│── dataset/
│ ├── with_mask/ # Training images with mask
│ └── without_mask/ # Training images without mask
│── examples/
│ └── sample.png # Example input image
│── README.md # Project documentation


---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/face_mask_detection.git
cd face_mask_detection

2. **Create a virtual environment (optional but recommended)**

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3. **Install dependencies**

pip install -r requirements.txt


✅ Requirements

Python 3.8+
OpenCV
TensorFlow / Keras
Numpy

📸 Example Output

Input image:

Output (prediction overlay):

✅ With Mask
❌ Without Mask
