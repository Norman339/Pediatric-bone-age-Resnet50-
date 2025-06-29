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

app = FastAPI(title="Bone Age Prediction API")

# Critical Hugging Face configuration
app.root_path = os.getenv("HF_HOME", "")

# Enhanced CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Device configuration (force CPU for Hugging Face)
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gender: int = 1
):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Only JPEG/PNG images accepted")
        
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(image_tensor, gender_tensor)
            bone_age = prediction.item()

        return JSONResponse({
            "bone_age_months": round(bone_age, 2),
            "bone_age_years": round(bone_age/12, 2),
            "gender": "male" if gender else "female"
        })

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/")
async def health_check(request: Request):
    return {
        "status": "running",
        "docs_url": f"{request.base_url}docs",
        "predict_url": f"{request.base_url}predict"
    }

# Hugging Face specific middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)