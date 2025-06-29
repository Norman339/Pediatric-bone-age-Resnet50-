from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import io
from torchvision import transforms
import numpy as np

app = FastAPI(title="Bone Age Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Add this
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture
class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, 128)
        self.fc = nn.Linear(128 + 1, 1)

    def forward(self, images, gender):
        x = self.resnet(images)
        gender = gender.view(-1, 1)
        x = torch.cat((x, gender), dim=1)
        return self.fc(x)

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Model loading with error handling
try:
    model = BoneAgeModel().to(device)
    checkpoint = torch.load('bone_age_res50_epoch_101.pth', map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # If checkpoint is just the state_dict
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError("Failed to load model") from e

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="X-ray image (JPEG/PNG)"),
    gender: int = 1  # 1 for male, 0 for female
):
    """Predict bone age from hand X-ray image"""
    try:
        # Validate input
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Only JPEG/PNG images accepted")
        
        if gender not in [0, 1]:
            raise HTTPException(400, "Gender must be 0 (female) or 1 (male)")

        # Process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            prediction = model(image_tensor, gender_tensor)
            bone_age = prediction.item()

        return {
            "bone_age_months": round(bone_age, 2),
            "bone_age_years": round(bone_age/12, 2),
            "gender": "male" if gender else "female",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    }
    # Required for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)