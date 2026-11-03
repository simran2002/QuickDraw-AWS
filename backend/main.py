import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io
from transformers import AutoModelForImageClassification, AutoImageProcessor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "./ai_model" 
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready!")

@app.get("/categories")
async def get_categories():
    labels = list(model.config.id2label.values())
    return {"categories": labels}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        image = ImageOps.contain(image, (224, 224))
        
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        prediction_idx = logits.argmax(-1).item()
        probs = F.softmax(logits, dim=-1)
        confidence = float(probs[0][prediction_idx].item() * 100)
        
        label = model.config.id2label[prediction_idx]

        return {
            "prediction": label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
