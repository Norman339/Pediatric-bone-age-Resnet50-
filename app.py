import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from torchvision import transforms

app = FastAPI(title="Bone Age Prediction API")

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, 128)
        self.fc = nn.Linear(128 + 1, 1)

    def forward(self, images, gender):
        x = self.resnet(images)
        gender = gender.view(-1, 1)
        x = torch.cat((x, gender), dim=1)
        return self.fc(x)

# Initialize model
model = BoneAgeModel()
model.load_state_dict(torch.load('bone_age_res50_epoch_101.pth', map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="X-ray image file"),
    gender: int = 1,  # 0 for female, 1 for male
):
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(image_tensor, gender_tensor)
            bone_age = prediction.item()

        return JSONResponse({
            "bone_age_months": round(bone_age, 2),
            "bone_age_years": round(bone_age/12, 2),
            "gender": "male" if gender else "female"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "device": str(device)}