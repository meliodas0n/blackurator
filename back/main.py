from diffusers import StableDiffusionPipeline
import torch




if __name__ == "__main__":
	model_id = "sd-legacy/stable-diffusion-v1-5"
	pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
	pipe = pipe.to("cuda")

	prompt = "A photo of an astronaut riding a horse on mars."
	images = pipe(prompt).images

	for index, img in enumerate(images):
		img.save(f"img{index}.jpg")
