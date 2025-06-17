# 🌿 Moringa Leaf Disease Detector with Grad-CAM

An AI-powered web application that detects and explains diseases in **Moringa leaves** using image classification and Grad-CAM heatmaps. This solution is designed to support smallholder farmers and agricultural researchers in identifying early signs of leaf diseases with visual explanation.

Built with **TensorFlow**, **Streamlit**, and a fine-tuned **EfficientNetB0** model, this tool classifies leaf images into four key categories and offers practical management tips based on the diagnosis.

---

## 🧠 What It Does

Upload a photo of a Moringa leaf, and the app will:

✅ Predict whether the leaf is:

- **Bacterial Leaf Spot**
- **Cercospora Leaf Spot**
- **Yellow Leaf (Nutrient/Water Stress)**
- **Healthy Leaf**

✅ Highlight the specific parts of the leaf that influenced the prediction using Grad-CAM heatmaps

✅ Provide disease information:
- Cause
- Symptoms
- Management practices

---

## 🚀 Try the App Live

👉 [https://moringadisease.streamlit.app](https://moringadisease.streamlit.app)

---

## 📁 Dataset Attribution

This project was made possible using the Moringa Leaf Disease dataset published on LinkedIn by:

- **Md. Al-Amin Sikder** 
- **Afia Noor**  
- **Md. Zaidul Islam**  
- **Firoz Mahmud**

Special thanks to the authors for making this valuable dataset publicly available for academic and development purposes.

---

## 🛠 Tech Stack

- Python 3.11  
- TensorFlow  
- Streamlit  
- tf-explain  
- OpenCV  
- NumPy  
- Pillow

---

## 💻 How to Run Locally

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/moringadis.git
   cd moringadis

2. (Optional) Create virtual environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


3. Install dependencies

pip install -r requirements.txt


4. Run the Streamlit app

streamlit run app.py




---

📦 Model Details

The app uses a fine-tuned EfficientNetB0 model saved in the native .keras format. The file is included in this repository (moringa_effnet_final.keras) and is automatically loaded at runtime.

Target size: 224x224 RGB

Classification output: 4 classes

Explainability: Grad-CAM using the block5c_project_conv layer



---

🎓 Project Context

This app was developed by Ayoola Mujib Ayodele as part of a real-world AI/ML project showcase within a training program. The goal is to explore how AI and deep learning can bridge the gap between modern technology and local agricultural challenges in Nigeria.


---

📌 Future Plans

Improve model performance with larger and cleaner datasets

Optimize the solution for mobile or offline use

Explore opportunities for academic publication based on feedback



---

📬 Contact

Feel free to reach out with suggestions, questions, or collaboration ideas:

📧 Email: ayodelemujibayoola@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/ayoolaayodelemujib




> "Empowering local agriculture through accessible AI."
