# 🍽️ Food Image Classifier - Deep Learning for Nutritional AI

This project is part of a broader initiative to build AI tools that assist with health and nutrition. In this module, I developed a deep learning model that classifies food images into 101 categories using the Food-101 dataset.

The goal is to demonstrate how computer vision can be applied to real-world use cases in healthtech and personalized nutrition.

---

## 🧠 Project Goals

- Train and evaluate a CNN-based food classifier
- Explore transfer learning with pretrained EfficientNet
- Deploy a reusable image classification pipeline
- Lay the groundwork for an AI system that connects food recognition to nutritional insights

---

## 🗂️ Project Structure

```text
food-vision/
├── data/                  # Food-101 dataset
├── notebooks/             # EDA + training experiments
├── src/                   # Training, evaluation, model code
├── outputs/               # Saved model weights and sample outputs
├── streamlit_app/         # (Optional) simple web demo
├── requirements.txt
└── README.md
```

## 📊 Dataset

Name:        Food-101  
Link:        https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/  
Classes:     101 food types (e.g., pizza, sushi, pad thai)  
Samples:     101,000 images (750 train / 250 test per class)  
Access via:  tensorflow_datasets or manual download  


## 🚀 How to Run
``` text
# 1. Clone the repo
git clone https://github.com/yourusername/food-vision.git
cd food-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Run this in a Python script or notebook
import tensorflow_datasets as tfds
tfds.load("food101", split='train', with_info=True, as_supervised=True)

# 4. Train the model

python src/train.py

# 5. Run the optional Streamlit demo

streamlit run streamlit_app/app.py
```

## 🧪 Model Performance

Model:           EfficientNetB0 (Transfer Learning)  
Top-1 Accuracy:  ~75% (on test set)  
Top-5 Accuracy:  Optional extension  
Loss Function:   Categorical Crossentropy  
Optimizer:       Adam  


## 📸 Sample Predictions

(Include visuals in outputs/sample_predictions/ or as a grid in your Streamlit app)


## 🔮 Future Plans

- Add OCR pipeline to extract nutrition facts from food packaging  
- Parse ingredients and nutrient information using NLP  
- Recommend dietary changes based on user health profiles  
- Wrap entire system into a health-focused AI recommendation engine  



## 🧰 Technologies Used

Languages:     Python 3.10+  
Libraries:     TensorFlow, TensorFlow Datasets, NumPy, Pandas, Matplotlib  
Modeling:      EfficientNet, CNNs, Transfer Learning  
Optional:      Streamlit (for web demo), Docker (for packaging)  


## 👤 Author

David Lord  
Deep Learning & Data Science Portfolio  
Email:     lordd007@gmail.com
GitHub:    https://github.com/Lordd007  
LinkedIn:  https://www.linkedin.com/in/david-lord-data-guy





