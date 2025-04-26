import streamlit as st
import cv2
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b4
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set page config
st.set_page_config(
    page_title="Cotton Leaf Analysis",
    page_icon="🌿",
    layout="wide"
)

# Define class names for disease classification
disease_class_names = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]

# Define deficiency labels and colors
deficiency_labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
deficiency_colors = ['#ff9999', '#99ff99', '#9999ff']

# Image transforms
disease_transform = A.Compose([
    A.Resize(380, 380),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

@st.cache_resource
def load_models():
    # Load disease classification model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disease_model = efficientnet_b4(weights=None)
    in_features = disease_model.classifier[1].in_features
    disease_model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 1024),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(1024),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(512),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, len(disease_class_names))
    )
    
    try:
        state_dict = torch.load("best_model.pth", map_location=device)
        disease_model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading disease model: {str(e)}")
        st.stop()
    
    disease_model.to(device)
    disease_model.eval()

    # Load deficiency prediction model
    try:
        deficiency_model = load_model("final_combined_model.h5")
    except Exception as e:
        st.error(f"Error loading deficiency model: {str(e)}")
        st.stop()

    return disease_model, deficiency_model, device

def preprocess_single_image(image, target_size=(128, 128)):
    img = Image.fromarray(image).resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(image, model, device):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = disease_transform(image=image_rgb)["image"]
    transformed = transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(transformed)
        pred = outputs.argmax(dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0, pred].item()
        all_probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return disease_class_names[pred], confidence, all_probs

def predict_deficiency(image, model):
    test_image = preprocess_single_image(image)
    raw_prediction = model.predict(test_image)[0]
    
    # Post-process prediction
    prediction = raw_prediction * 100 if np.max(raw_prediction) <= 1 else raw_prediction
    prediction = np.clip(prediction, 0, 100)
    
    # Normalize if total > 100%
    if prediction.sum() > 100:
        prediction = (prediction / prediction.sum()) * 100
        
    return prediction

def main():
    st.title("🌿 Cotton Leaf Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Disease Classification", "Nutrient Deficiency Analysis"])
    
    # Load models
    disease_model, deficiency_model, device = load_models()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the image
        image = np.array(Image.open(uploaded_file))
        
        with tab1:
            st.header("Disease Classification")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Make disease prediction
            predicted_class, confidence, all_probs = predict_disease(image_cv, disease_model, device)
            
            # Display disease results
            st.subheader("Prediction Results")
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            
            # Display probability distribution
            st.subheader("Probability Distribution")
            for i, (class_name, prob) in enumerate(zip(disease_class_names, all_probs)):
                st.progress(float(prob), text=f"{class_name}: {prob * 100:.2f}%")
        
        with tab2:
            st.header("Nutrient Deficiency Analysis")
            
            # Create figure for visualization
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            
            # Original Image
            ax_img = fig.add_subplot(gs[0])
            ax_img.imshow(image)
            ax_img.set_title("Original Image", fontsize=14)
            ax_img.axis('off')
            
            # Make deficiency prediction
            prediction = predict_deficiency(image, deficiency_model)
            
            # Bar Plot of Predictions
            ax_bar = fig.add_subplot(gs[1])
            bars = ax_bar.bar(deficiency_labels, prediction, color=deficiency_colors)
            ax_bar.set_ylim(0, 100)
            ax_bar.set_ylabel("Deficiency %", fontsize=12)
            ax_bar.set_title("Predicted Deficiency Percentages", fontsize=14)
            
            # Add percentage labels on bars
            for bar, value in zip(bars, prediction):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}%",
                          ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main() 