from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = load_model("mask_img_analyzer.h5")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    img_data = data["image"].split(",")[1]  # remove data:image/jpeg;base64,
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Preprocess image same as training
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)
    confidence = prediction[0][pred_label]
    label = "Mask" if pred_label == 1 else "No Mask"

    return JSONResponse({"label": label, "confidence": float(confidence)})
