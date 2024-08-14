import torch
import gradio as gr
import numpy as np
import random
import os
from datetime import datetime
from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline, AnimateDiffPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from diffusers import DDIMScheduler, MotionAdapter
from PIL import Image

# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Generate video using different models
def generate_video(model_name, prompt, image, resolution_x, resolution_y, negative_prompt, num_inference_steps, guidance_scale, num_frames, fps, seed, optimize_memory, optimize_speed):
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

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Apply optimizations
    def apply_optimizations(pipeline):
        if optimize_memory:
            pipeline.enable_model_cpu_offload()
            pipeline.unet.enable_forward_chunking()
        elif optimize_speed:
            pipeline.to("cuda")
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

    if model_name == "Stable Video Diffusion":
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        apply_optimizations(pipeline)

        generator = torch.manual_seed(seed)
        frames = pipeline(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
        video_path = f"output/{model_name.replace(' ', '_').lower()}_{timestamp}.mp4"
        export_to_video(frames, video_path, fps=fps)

    elif model_name == "I2VGen-XL":
        pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        apply_optimizations(pipeline)

        generator = torch.manual_seed(seed)

        frames = pipeline(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        video_path = f"output/{model_name.replace(' ', '_').lower()}_{timestamp}.gif"
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
        apply_optimizations(pipeline)

        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        frames = output.frames[0]
        video_path = f"output/{model_name.replace(' ', '_').lower()}_{timestamp}.gif"
        export_to_gif(frames, video_path)

    elif model_name == "ModelscopeT2V":
        pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        apply_optimizations(pipeline)

        video_frames = pipeline(prompt).frames[0]
        video_path = f"output/{model_name.replace(' ', '_').lower()}_{timestamp}.mp4"
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
            num_inference_steps = gr.Slider(0, 100, value=20, step=1, label="Number of Inference Steps")
            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
            num_frames = gr.Slider(1, 60, value=16, step=1, label="Number of Frames (for AnimateDiff)")
            fps = gr.Slider(1, 60, value=7, step=1, label="FPS")

            seed = gr.Number(value=-1, label="Seed", precision=0)
        
        # Optimization options
        with gr.Row():
            optimize_memory = gr.Checkbox(label="Optimize for Memory", value=False)
            optimize_speed = gr.Checkbox(label="Optimize for Speed (requires CUDA)", value=False)

        output = gr.Video(label="Generated Video")
        
        generate_btn = gr.Button("Generate Video")
        generate_btn.click(
            fn=generate_video,
            inputs=[model_name, prompt, image_input, resolution_x, resolution_y, negative_prompt, num_inference_steps, guidance_scale, num_frames, fps, seed, optimize_memory, optimize_speed],
            outputs=output
        )

    return iface

# Start the interface
if __name__ == "__main__":
    iface = get_interface()
    iface.launch(share=True)
