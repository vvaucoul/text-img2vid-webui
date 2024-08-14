# Video Generation with Diffusion Models

This project provides a web-based interface to generate videos using various diffusion models. The interface is built using Gradio and allows users to generate videos from text or images using state-of-the-art diffusion models.

## Features

- **Model Selection**: Choose from different models like Stable Video Diffusion, I2VGen-XL, AnimateDiff, and ModelscopeT2V.
- **Input Options**: Depending on the model selected, you can input either text prompts or images.
- **Customizable Parameters**: Adjust various settings such as resolution, number of inference steps, guidance scale, number of frames, FPS, and seed for reproducibility.
- **Video Output**: The generated videos are provided as either GIF or MP4 files, depending on the selected model.

## Requirements

- Python 3.8 or higher
- `torch` (PyTorch)
- `diffusers` (for diffusion models)
- `gradio` (for the web interface)
- `Pillow` (for image processing)

You can install the necessary Python packages using the following command:

```bash
pip install torch diffusers gradio Pillow
```

## Usage

### Running the Application

To run the application locally, execute the following command:

```bash
python app.py
```

This will start a local Gradio interface. If you want to share the interface with others, the application will provide a shareable link.

### Deploying the Application

For free permanent hosting and GPU upgrades, you can deploy the application to Hugging Face Spaces using the following command:

```bash
gradio deploy
```

### Interface Overview

Once the application is running, you will see the following options:

- Model Selection: Choose the diffusion model to use.
- Prompt Input: (Text-to-Video models) Enter a text prompt describing the video you want to generate.
- Image Input: (Image-to-Video models) Upload an image to be used as the starting point for the video.
- Resolution: Adjust the resolution (width and height) of the output video.
- Negative Prompt: (Optional) Enter negative prompts to avoid certain aspects in the video.
- Number of Inference Steps: Set the number of steps the model should use for inference.
Guidance Scale: Adjust the guidance scale to control the influence of the text/image prompt on the output.
- Number of Frames: (AnimateDiff only) Set the number of frames in the generated video.
FPS: Set the frames per second for the output video.
- Seed: Set a seed for reproducibility of the results.
Generating a Video
- Select a Model: Choose the desired diffusion model from the dropdown.
- Enter Prompt or Upload Image: Depending on the model, enter a prompt or upload an image.
- Adjust Parameters: Customize the resolution, inference steps, guidance scale, etc., according to your requirements.
- Click 'Generate Video': The model will process the input and generate a video, which will be displayed on the interface.
- Download the Video: You can download the generated video from the interface.

## Example

Here's an example of how you might use the interface:

- Model: AnimateDiff
- Prompt: "A space rocket launching into space"
- Resolution: 1024x576
- FPS: 7
- Seed: 42

After adjusting these settings, click on "Generate Video" to see the result.

### Troubleshooting
If you encounter any issues, make sure that all dependencies are installed correctly and that your Python environment is properly set up.

### Common Issues
AttributeError for ANTIALIAS: Ensure that you are using Image.Resampling.LANCZOS instead of Image.ANTIALIAS in recent versions of Pillow.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
The diffusion models used in this project are from the Hugging Face Diffusers library.
Gradio is used for the web-based user interface.
