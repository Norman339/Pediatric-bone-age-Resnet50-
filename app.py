#!/usr/bin/env python3
"""
Bone Age Predictor - Hugging Face Spaces Optimized Version
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import sys

print("🚀 Starting Bone Age Predictor for Hugging Face Spaces...")

# Force CPU usage for Hugging Face Spaces
device = torch.device("cpu")
print(f"📱 Using device: {device}")

class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, 128)
        self.fc = nn.Linear(128 + 1, 1)
    
    def forward(self, x, gender):
        features = self.resnet(x)
        combined = torch.cat([features, gender.unsqueeze(1)], dim=1)
        return self.fc(combined)

# Load model with better error handling
model = None
try:
    print("🔄 Loading model...")
    model = BoneAgeModel().to(device)
    
    # Check multiple possible model paths
    possible_paths = [
        'bone_age_res50_epoch_101.pth',
        '../bone_age_res50_epoch_101.pth',
        './bone_age_res50_epoch_101.pth'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Model file not found in any expected location")
    
    print(f"📁 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    model = None

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_bone_age(image, gender):
    """Predict bone age from X-ray image"""
    if model is None:
        return "❌ Model not loaded. Please check if the model file is available."
    
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor, gender_tensor)
            bone_age = prediction.item()
            
        return f"Predicted Bone Age: {bone_age:.1f} years"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
print("🎨 Creating Gradio interface...")

with gr.Blocks(title="Bone Age Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦴 Pediatric Bone Age Predictor")
    gr.Markdown("Upload an X-ray image of a child's hand/wrist to predict their bone age.")
    
    if model is None:
        gr.Markdown("⚠️ **Model loading failed. Please check the model file.**")
    else:
        gr.Markdown("✅ **Model loaded successfully! Ready for predictions.**")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload X-ray Image", type="pil")
            gender = gr.Radio(
                choices=["Female", "Male"], 
                label="Gender", 
                value="Male"
            )
            predict_btn = gr.Button("Predict Bone Age", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Prediction Result", lines=3)
    
    gr.Markdown("""
    ### Instructions:
    1. Upload a clear X-ray image of the child's hand/wrist
    2. Select the child's gender
    3. Click 'Predict Bone Age' to get the result
    
    ### Note:
    This tool is for educational/research purposes. Always consult with medical professionals for clinical decisions.
    """)
    
    predict_btn.click(
        fn=lambda img, g: predict_bone_age(img, 1 if g == "Male" else 0),
        inputs=[input_image, gender],
        outputs=output_text,
        show_progress=True
    )

print("🚀 Launching app...")

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)