import os

import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_image(self, prompt: str, image_path: str):
        # Verifica se o diretório existe, caso contrário, cria-o
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.pipeline.safety_checker = lambda images, **kwargs: (images, False)
        # Gera a imagem e salva no caminho especificado
        image = self.pipeline(prompt).images[0]
        image.save(image_path)
        return image_path
