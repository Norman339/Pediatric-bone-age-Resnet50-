---
title: Bone Age Predictor
emoji: 🐨
colorFrom: yellow
colorTo: pink
sdk: docker
pinned: false
license: other
short_description: Pediatric bone age
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🦴 Pediatric Bone Age Predictor

A deep learning model that predicts bone age from pediatric X-ray images using ResNet-50 architecture.

## Features

- **AI-Powered Analysis**: Uses a trained ResNet-50 model to analyze hand/wrist X-rays
- **Gender-Aware Prediction**: Takes into account gender differences in bone development
- **User-Friendly Interface**: Simple Gradio web interface for easy interaction
- **Medical Grade**: Designed for pediatric bone age assessment

## How to Use

1. Upload a clear X-ray image of a child's hand/wrist
2. Select the child's gender (Male/Female)
3. Click "Predict Bone Age" to get the result
4. The system will return the predicted bone age in years

## Technical Details

- **Model**: Custom ResNet-50 with additional regression layers
- **Input**: X-ray images (224x224 pixels) + gender information
- **Output**: Predicted bone age in years
- **Framework**: PyTorch with Gradio interface

## Deployment

### For Hugging Face Spaces

1. Ensure the model file `bone_age_res50_epoch_101.pth` is in the repository
2. The app will automatically load the model on startup
3. Deploy to Hugging Face Spaces using the provided Dockerfile

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Important Notes

⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. Always consult with qualified medical professionals for clinical decisions.

## Model File

The model requires the trained weights file `bone_age_res50_epoch_101.pth`. Make sure this file is available in the repository for the app to function properly.

## Troubleshooting

If you see "Model not loaded" error:
1. Check that the model file exists in the repository
2. Ensure the file path is correct
3. Verify the model file is not corrupted

---

Built with ❤️ for pediatric healthcare research
