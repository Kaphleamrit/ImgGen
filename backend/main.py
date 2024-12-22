from fastapi import HTTPException, Query, Request
import modal

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]==0.115.4", "diffusers", "transformers", "accelerate")
app = modal.App(name="has-simple-web-endpoint", image=image)

with image.imports():
    from diffusers import AutoPipelineForText2Image
    import torch
    import io
    from fastapi import Response
    import os
    from datetime import datetime, timezone


@app.cls(image=image, gpu="A10G",
         container_idle_timeout=300,
         secrets=[modal.Secret.from_name("secret")])
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]

    @modal.web_endpoint()
    def genImage(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):

        api_key = request.headers.get("X_API_KEY")

        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized Access"
            )
        image = self.pipe(prompt=prompt, num_inference_steps=1,
                          guidance_scale=0.0).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return Response(content=buffer.getvalue(), media_type="image/jpeg")


@modal.web_endpoint()
def health(self):
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.function()
def square(x):
    return x**2

 
# run every 5 minutes
@app.function(schedule=modal.Cron("*/5 * * * *"),
              secrets=[modal.Secret.from_name("secret")])
def update_keep_warm():
    health_url = "https://kaphleamrit--has-simple-web-endpoint-model-genimage.modal.run"
    generate_url = " https://kaphleamrit--has-simple-web-endpoint-model-genimage.modal.run"
    peak_hours_start, peak_hours_end = 6, 18
    if peak_hours_start <= datetime.now(timezone.utc).hour < peak_hours_end:
        square.keep_warm(3)
    else:
        square.keep_warm(0)
