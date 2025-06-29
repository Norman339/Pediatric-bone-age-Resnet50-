import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os
import time
import threading

# Device configuration (must use CPU for Hugging Face Spaces)
device = torch.device("cpu")

class BoneAgeModel(nn.Module):
    def __init__(self, use_simple_model=False):
        super().__init__()
        if use_simple_model:
            # Use ResNet-18 for faster inference
            self.resnet = models.resnet18(weights=None)
            self.resnet.fc = nn.Linear(512, 128)
        else:
            # Use ResNet-50 (original)
            self.resnet = models.resnet50(weights=None)
            self.resnet.fc = nn.Linear(2048, 128)
        
        self.fc = nn.Linear(128 + 1, 1)
    
    def forward(self, x, gender):
        features = self.resnet(x)
        combined = torch.cat([features, gender.unsqueeze(1)], dim=1)
        return self.fc(combined)

# Load model
model = None
try:
    print("🔄 Loading model...")
    model = BoneAgeModel().to(device)
    
    # Check if model file exists in current directory or parent directory
    model_path = 'bone_age_res50_epoch_101.pth'
    if not os.path.exists(model_path):
        model_path = '../bone_age_res50_epoch_101.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"📁 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
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
    """
    Predict bone age from X-ray image
    
    Args:
        image: PIL Image or numpy array
        gender: 0 for female, 1 for male
    
    Returns:
        Predicted bone age in years
    """
    if model is None:
        return "❌ Model not loaded. Please check if the model file is available."
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Make prediction with timeout
        result = [None]
        error = [None]
        
        def run_prediction():
            try:
                with torch.no_grad():
                    prediction = model(image_tensor, gender_tensor)
                    bone_age = prediction.item()
                    result[0] = f"Predicted Bone Age: {bone_age:.1f} years"
            except Exception as e:
                error[0] = str(e)
        
        # Run prediction with 30-second timeout
        thread = threading.Thread(target=run_prediction)
        thread.start()
        thread.join(timeout=30)
        
        if thread.is_alive():
            return "❌ Prediction timed out (took too long). Try with a smaller image or restart the app."
        
        if error[0]:
            if "out of memory" in error[0].lower():
                return "❌ Error: Out of memory. Try with a smaller image or restart the app."
            else:
                return f"❌ Error: {error[0]}"
        
        # Clear tensors from memory
        del image_tensor, gender_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result[0]
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Bone Age Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦴 Pediatric Bone Age Predictor")
    gr.Markdown("Upload an X-ray image of a child's hand/wrist to predict their bone age.")
    
    # Add loading message
    if model is None:
        gr.Markdown("⚠️ **Loading model... Please wait a moment for the first prediction.**")
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
    
    # Add some helpful information
    gr.Markdown("""
    ### Instructions:
    1. Upload a clear X-ray image of the child's hand/wrist
    2. Select the child's gender
    3. Click 'Predict Bone Age' to get the result
    
    ### Note:
    - First prediction may take 2-3 minutes as the model loads
    - Subsequent predictions will be faster
    - This tool is for educational/research purposes. Always consult with medical professionals for clinical decisions.
    """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=lambda img, g: predict_bone_age(img, 1 if g == "Male" else 0),
        inputs=[input_image, gender],
        outputs=output_text,
        show_progress=True
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 