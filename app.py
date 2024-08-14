import torch
import gradio as gr
import numpy as np
import random
from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline, AnimateDiffPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from diffusers import DDIMScheduler, MotionAdapter
from PIL import Image

# Generate video using different models
def generate_video(model_name, prompt, image, resolution_x, resolution_y, negative_prompt, num_inference_steps, guidance_scale, num_frames, fps, seed):
    # Use random seed if selected
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    
    # Check if image is a path or a numpy array
    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Image resizing
    image = image.resize((resolution_x, resolution_y), Image.Resampling.LANCZOS)

    if model_name == "Stable Video Diffusion":
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline.enable_model_cpu_offload()

        generator = torch.manual_seed(seed)
        frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
        video_path = "generated_stable.mp4"
        export_to_video(frames, video_path, fps=fps)

    elif model_name == "I2VGen-XL":
        pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        pipeline.enable_model_cpu_offload()

        generator = torch.manual_seed(seed)

        frames = pipeline(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        video_path = "generated_i2v.gif"
        export_to_gif(frames, video_path)

    elif model_name == "AnimateDiff":
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
        pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)

        scheduler = DDIMScheduler.from_pretrained(
            "emilianJR/epiCRealism",
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipeline.scheduler = scheduler
        pipeline.enable_vae_slicing()
        pipeline.enable_model_cpu_offload()

        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        frames = output.frames[0]
        video_path = "generated_animate.gif"
        export_to_gif(frames, video_path)

    elif model_name == "ModelscopeT2V":
        pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()

        video_frames = pipeline(prompt).frames[0]
        video_path = "generated_modelscope.mp4"
        export_to_video(video_frames, video_path, fps=fps)

    return video_path

# Gradio interface
def get_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Video Generation Web UI")
        gr.Markdown("This web UI allows you to generate videos using different models. Select a model, enter the prompt or upload an image, and click on the 'Generate Video' button to generate the video.")
        gr.Markdown("Made by [Vintz](https://github.com/vvaucoul) with ❤️ by [Huggingface](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid)")
        
        
        with gr.Row():
            model_name = gr.Dropdown(
                choices=["Stable Video Diffusion", "I2VGen-XL", "AnimateDiff", "ModelscopeT2V"], 
                label="Select Model",
                value="Stable Video Diffusion"
            )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt (for text-to-video models)", placeholder="Enter your prompt here")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")

            image_input = gr.Image(label="Image (for image-to-video models)")
        
        with gr.Row():
            resolution_x = gr.Slider(256, 2048, value=1024, step=32, label="Resolution X")
            resolution_y = gr.Slider(256, 2048, value=1024, step=32, label="Resolution Y")
        
        # Parameters for video generation
        with gr.Column():
            num_inference_steps = gr.Slider(10, 150, value=50, step=5, label="Number of Inference Steps")
            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
            num_frames = gr.Slider(1, 32, value=16, step=1, label="Number of Frames (for AnimateDiff)")
            fps = gr.Slider(1, 60, value=7, step=1, label="FPS")

            seed = gr.Number(value=-1, label="Seed", precision=0)
        
        output = gr.Video(label="Generated Video")
        
        generate_btn = gr.Button("Generate Video")
        generate_btn.click(
            fn=generate_video,
            inputs=[model_name, prompt, image_input, resolution_x, resolution_y, negative_prompt, num_inference_steps, guidance_scale, num_frames, fps, seed],
            outputs=output
        )

    return iface

# Start the interface
if __name__ == "__main__":
    iface = get_interface()
    iface.launch(share=True)
