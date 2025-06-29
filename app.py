from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import io
from torchvision import transforms
import os

# Critical initialization for Hugging Face
app = FastAPI()

# Required for Hugging Face Spaces
app.root_path = os.getenv("HF_HOME", "")

# Device configuration (must use CPU)
device = torch.device("cpu")

class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, 128)
        self.fc = nn.Linear(128 + 1, 1)

# Load model
try:
    model = BoneAgeModel().to(device)
    checkpoint = torch.load('bone_age_res50_epoch_101.pth', map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Load only the model weights if checkpoint contains training metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        # Alternative common format
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the file is just the state_dict
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError("Failed to load model") from e

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), gender: int = 1):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor, gender_tensor)
            bone_age = prediction.item()
            
        return {"bone_age": bone_age}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "API is running"}

# Required Hugging Face middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response