from time import strftime

from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import autocast, torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, revision="fp16"
)
pipe = pipe.to(device)


def disable_nsfw_filter(images, **kwargs):
    return images, False


pipe.safety_checker = disable_nsfw_filter


@app.get("/")
def generate(prompt: str):

    with autocast(device):
        image = pipe(prompt, num_inference_steps=50, height=320, width=320).images[0]

    image.save(str(strftime("%Y%m%d-%H%M%S")) + ".png")

    return {"Result": "Succeded!"}
