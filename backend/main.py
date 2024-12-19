import modal

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]==0.115.4", "diffusers", "transformers", "accelerate").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
app = modal.App(name="has-simple-web-endpoint", image=image)

with image.imports():
    from diffusers import AutoPipelineForText2Image
    import torch
    import io
    from fastapi import Response


@app.cls(image=image, gpu="A10G")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

    @modal.web_endpoint()
    def genImage(self, prompt="Superman working hard in the paddy field"):
        image = self.pipe(prompt=prompt, num_inference_steps=1,
                          guidance_scale=0.0).images[0]
        

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
