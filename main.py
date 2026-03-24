from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import tempfile
import os

app = FastAPI()

# Allows requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best-better.pt")

# Converts frames to base64, easy format for computer vision to understand
def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

# Runs model on every frame
def run_model_on_frame(frame):
    results = model(frame, stream=False)
    for r in results:
        annotated = r.plot()
        return annotated
    return frame


# If user uploads an image 
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    #Error handling if image can't be decoded
    if frame is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)

    annotated = run_model_on_frame(frame)
    img_b64 = frame_to_base64(annotated)

    return {"image": img_b64}


# If user uploads a video, process every 10th frame to maintain speed
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()

    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames_b64 = []
    frame_count = 0
    sample_every = 10  # Process every 10th frame to keep response fast

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            annotated = run_model_on_frame(frame)
            frames_b64.append(frame_to_base64(annotated))
        frame_count += 1

    cap.release()
    os.remove(tmp_path)

    if not frames_b64:
        return JSONResponse({"error": "Could not process video."}, status_code=400)

    return {"frames": frames_b64, "total": len(frames_b64)}


# If user chooses Webcam feed
@app.post("/detect/webcam")
async def detect_webcam(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Could not decode frame."}, status_code=400)

    annotated = run_model_on_frame(frame)
    img_b64 = frame_to_base64(annotated)

    return {"image": img_b64}


# Endpoint to confirm if API is running
@app.get("/")
def root():
    return {"status": "Human Detection API is running."}