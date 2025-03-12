# FastAPI Server for Image Generation

This FastAPI service generates images based on text prompts using a locally stored model.

## **Installation**
Ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:
```txt
torch 
torchvision
diffusers==0.28.0
transformers==4.41.1
gradio==4.31.5
bitsandbytes==0.43.1
accelerate==0.30.1
protobuf==3.20
opencv-python
tensorboardX
safetensors
pillow
einops
torch
peft==0.10.0
fastapi 
uvicorn
```

## **Running the FastAPI Server**
To start the FastAPI server, use:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## **Sending Requests**
Once the server is running, you can send a request using `curl`:
```bash
curl -X 'POST' 'http://127.0.0.1:8000/generate' \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "1girl, tifa lockhart, final fantasy", "model_path": "./LayerDiffuse_DiffusersCLI"}' \
     --output output.png
```

## **Expected Output**
- If successful, `output.png` will contain the generated image.
- If there are errors, ensure that:
  - The model files exist in the correct path.
  - All dependencies are installed.
  - The FastAPI server is running properly.

