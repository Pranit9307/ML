# Cotton Leaf Disease Detection and Nutrient Deficiency Estimation

### 🔗 Project Links
- [Efficient-Net-B4 Classification Results][https://www.kaggle.com/code/pranit9/ml-project-new]
---
- [Preprocessing and Regression Model Results][https://colab.research.google.com/drive/1eQ3oQnw8qT-Owl2KY_HbiniahzEKETeb?usp=sharing]
---
- [Dataset Used][https://data.mendeley.com/datasets/b3jy2p6k8w/2]
---

> _(Replace `#` above with actual URLs when available.)_

---

## 📜 Overview

This project presents a Deep Learning-based pipeline for:
- Detecting **cotton leaf diseases** (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy),
- Quantifying **nutrient deficiencies** (Nitrogen, Phosphorus, Potassium) from cotton leaf images.

It integrates **image preprocessing**, **EfficientNet-B4 based disease classification**, **color-based nutrient deficiency detection**, and **regression modeling** to offer detailed insights for agricultural monitoring.

---

## 🛠️ Project Structure

- **Color Enhancement:**  
  Enhances input images using adaptive histogram-based contrast enhancement to highlight leaf features.

- **Disease Classification (EfficientNet-B4):**  
  A fine-tuned EfficientNet-B4 model identifies the disease category from enhanced leaf images.

- **Nutrient Deficiency Detection:**  
  Analyzes color features to detect and quantify Nitrogen (N), Phosphorus (P), and Potassium (K) deficiencies.

- **Predictive Modeling:**  
  Trains a regression model using extracted color features to estimate severity percentages of each nutrient deficiency.

- **Final Report:**  
  Provides a diagnosis (disease label) and nutrient deficiency percentages in a structured format.

---

## 📊 Technologies Used

- **Deep Learning:** TensorFlow / Keras (EfficientNet-B4)
- **Machine Learning:** Scikit-Learn (Regression models)
- **Image Processing:** OpenCV (Color space transformations: HSV, Lab)
- **Data Handling:** Pandas, CSV data storage
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Dataset

The dataset includes:
- High-quality images of cotton leaves labeled by disease type,
- Manually verified nutrient deficiency levels for supervised training.


---

## 📈 Results

- **EfficientNet-B4 Classification Results:**
  - High accuracy achieved in disease classification.
  - Use of data augmentation and fine-tuning improved generalization.

- **Regression Model Results:**
  - Accurate estimation of Nitrogen, Phosphorus, and Potassium deficiencies.
  - Enabled actionable feedback for crop health management.



---

## 🔮 Future Scope

- Extend nutrient deficiency detection to include secondary and micronutrients.
- Explore hyperspectral imaging for better feature extraction.
- Develop real-time mobile/web applications for farmers.
- Implement Explainable AI (XAI) methods for better transparency and trust.

---

## 👨‍💻 Authors

- Mansi Rokade
- Sejal Rokade
- Pranav Sakpal
- Pranit Sarode

(Pimpri Chinchwad College Of Engineering, Pune - Department of Computer Engineering)

---
