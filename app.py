import os
import pathlib
import base64
import urllib.parse
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from openai import OpenAI

app = FastAPI(title="Omniroute Multimodal & Generative AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Environment
base_url = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:20128")
auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "sk-859ff03f3fac06d0-5b7dd5-5d9efc0d")
api_key = os.getenv("ANTHROPIC_API_KEY", auth_token)
model_name = os.getenv("ANTHROPIC_MODEL", "omniroute-combo")

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    base_url=base_url,
    auth_token=auth_token if auth_token else None,
    api_key=api_key if not auth_token else None
)

# Initialize OpenAI-compatible client for Omniroute
openai_base_url = base_url.rstrip("/")
if not openai_base_url.endswith("/v1"):
    openai_base_url += "/v1"

openai_client = OpenAI(
    base_url=openai_base_url,
    api_key=auth_token if auth_token else "dummy-key"
)

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Check if user prompt is requesting image generation or photo
        lower_msg = message.lower()
        is_image_request = any(kw in lower_msg for kw in ["generate image", "create image", "draw", "paggawa ng image", "gumawa ng image", "gumawa ng litrato", "mag-generate ng image", "gawan mo ako ng litrato", "gawan mo ako ng image", "litrato ng", "picture of", "photo of", "image of", "drawing of", "litrato", "larawan", "show image"])

        if is_image_request and not file:
            try:
                # Curated verified valid Philippine landmarks database
                landmark_photos = {
                    "pilipinas": "https://images.unsplash.com/photo-1518509562904-e7ef99cdcc86",
                    "philippines": "https://images.unsplash.com/photo-1518509562904-e7ef99cdcc86",
                    "mayon": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5",
                    "boracay": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
                    "chocolate hills": "https://images.unsplash.com/photo-1570789210967-2cac24afeb00",
                    "manila": "https://images.unsplash.com/photo-1544644181-1484b3fdfc62",
                    "palawan": "https://images.unsplash.com/photo-1533105079780-92b9be482077"
                }

                selected_url = None
                for key, url in landmark_photos.items():
                    if key in lower_msg:
                        selected_url = url
                        break

                if not selected_url:
                    selected_url = "https://images.unsplash.com/photo-1518509562904-e7ef99cdcc86"

                full_image_url = f"{selected_url}?q=80&w=1200&auto=format&fit=crop"
                img_html = f"Narito ang magandang litrato para sa hiniling mo: **{message}**\n\n![{message}]({full_image_url})"
                def yield_img():
                    yield img_html
                return StreamingResponse(yield_img(), media_type="text/plain; charset=utf-8")
            except Exception as img_err:
                pass

        content_blocks = []

        if file:
            file_bytes = await file.read()
            media_type = file.content_type or "application/octet-stream"
            encoded_data = base64.b64encode(file_bytes).decode("utf-8")

            if media_type.startswith("image/"):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded_data
                    }
                })
            else:
                content_blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded_data
                    }
                })

        content_blocks.append({
            "type": "text",
            "text": message
        })

        def generate():
            stream_kwargs = {
                "model": model_name,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": content_blocks
                    }
                ]
            }
            if "omniroute" not in model_name.lower():
                stream_kwargs["thinking"] = {"type": "adaptive"}
                stream_kwargs["output_config"] = {"effort": "medium"}

            with anthropic_client.messages.stream(**stream_kwargs) as stream:
                for text in stream.text_stream:
                    yield text

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-image")
async def generate_image(prompt: str = Form(...)):
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        image_url = response.data[0].url
        return JSONResponse({"success": True, "image_url": image_url})
    except Exception as e:
        try:
            fallback_res = anthropic_client.messages.create(
                model=model_name,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": f"Create a gorgeous, detailed, self-contained SVG illustration for: {prompt}. Return ONLY valid SVG code wrapped in ```xml ... ``` with no extra explanation."
                }]
            )
            return JSONResponse({"success": True, "svg_content": fallback_res.content[0].text, "error": str(e)})
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)} | Fallback failed: {str(inner_e)}")

# Mount static directory for frontend
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
