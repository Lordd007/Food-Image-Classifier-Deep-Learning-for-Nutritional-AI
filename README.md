# 🍽️ Food Image Classifier - Deep Learning for Nutritional AI

This project is part of a broader initiative to build AI tools that assist with health and nutrition. In this module, I developed a deep learning model that classifies food images into 101 categories using the Food-101 dataset.

The goal is to demonstrate how computer vision can be applied to real-world use cases in healthtech and personalized nutrition.

---

## 🧐 Project Goals

* Train and evaluate a CNN-based food classifier
* Explore transfer learning with pretrained EfficientNet and MobileNetV2
* Evaluate performance using Top1/Top-5 accuracy, confusion matrix, and precision/recall
* Deploy a Gradio UI to interactively test predictions
* Lay the groundwork for a system that connects food recognition to nutritional insights

---

## 🗂️ Project Structure

```text
food-vision/
├── notebooks/             # Training, evaluation, and visualization notebooks
├── gradio_app.py          # Simple interactive demo using Gradio
├── requirements.txt       # Dependency list
├── README.md              # Project overview and instructions
```

## 📊 Dataset

* **Name**: Food-101
* **Link**: [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* **Access**: `tensorflow_datasets`
* **Classes**: 101 food types (e.g., pizza, sushi, pad thai)
* **Size**: 101,000 images (750 train / 250 test per class)

```python
import tensorflow_datasets as tfds
(train_ds, val_ds), ds_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True
)
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Lordd007/Food-Image-Classifier-Deep-Learning-for-Nutritional-AI.git
cd food-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run model training (in Jupyter or Colab)
jupyter notebook  # or open Food101_Training.ipynb in Colab

# 4. Launch interactive Gradio demo (optional)
python gradio_app.py  # or run in notebook with share=True

```

---

## 📊 Model Performance

* **Base Model**: EfficientNetB0 (also tested MobileNetV2)
* **Fine-tuned**: Yes (after initial training)
* **Top-1 Accuracy**: \~58.5%
* **Top-5 Accuracy**: \~83%
* **Loss Function**: Sparse Categorical Crossentropy
* **Optimizer**: Adam
* **Evaluation**: Confusion matrix, classification report, precision/recall per class

---

## 📸 Gradio Demo

Upload a food image and get:

* Top-5 predicted labels with confidence
* The predicted label overlaid on the image

> Run with: `demo.launch(share=True)` in the notebook or script.

---

## 🔮 Future Plans

* OCR pipeline to extract nutrition facts from packaging
* NLP analysis to parse ingredients and macro/micronutrients
* Integrate user health profiles for dietary recommendations
* Build a recommendation engine using image + text inputs
* Consider mobile deployment with TensorFlow Lite

---

## 🛠️ Tech Stack

* **Languages**: Python 3.10+
* **Libraries**: TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
* **Visualization**: Gradio, Seaborn heatmaps, bar plots
* **Deployment**: Gradio (with option for Hugging Face Spaces)

---

## 👤 Author

**David Lord**
Deep Learning & Data Science Portfolio
**Email**: [lordd007@gmail.com](mailto:lordd007@gmail.com)
**GitHub**: [github.com/Lordd007](https://github.com/Lordd007)
**LinkedIn**: [linkedin.com/in/david-lord-data-guy](https://www.linkedin.com/in/david-lord-data-guy)

---

Feel free to open an issue or contact me if you'd like to collaborate or learn more!
