from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import io

app = FastAPI(title="Bone Age Prediction API")

# Enable CORS (critical for Hugging Face)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, 128)
        self.fc = nn.Linear(128 + 1, 1)

model = BoneAgeModel().to(device)
model.load_state_dict(torch.load('bone_age_res50_epoch_101.pth', map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...), gender: int = 1):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Add your image preprocessing here
        return {"bone_age": 144.2, "units": "months"}  # Mock response
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "device": str(device)}