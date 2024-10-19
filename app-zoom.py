import gradio as gr
import torch
import devicetorch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw
import numpy as np
import cv2
import tempfile
import os

DEVICE = devicetorch.get(torch)

OUTPUT_DIR = "outputs" 


# Load models and configurations
config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device=DEVICE, dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(DEVICE)

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to(DEVICE)

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def infer(image, width=1024, height=1024, overlap_width=18, num_inference_steps=8, resize_option="custom", custom_resize_size=768, prompt_input=None, alignment="Middle"):
    source = image
    target_size = (width, height)
    overlap = overlap_width

    # Upscale if source is smaller than target in both dimensions
    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

    if source.width > target_size[0] or source.height > target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)
    
    if resize_option == "Full":
        resize_size = max(source.width, source.height)
    elif resize_option == "1/2":
        resize_size = max(source.width, source.height) // 2
    elif resize_option == "1/3":
        resize_size = max(source.width, source.height) // 3
    elif resize_option == "1/4":
        resize_size = max(source.width, source.height) // 4
    else:  # Custom
        resize_size = custom_resize_size

    aspect_ratio = source.height / source.width
    new_width = resize_size
    new_height = int(resize_size * aspect_ratio)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    if not can_expand(source.width, source.height, target_size[0], target_size[1], alignment):
        alignment = "Middle"

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - source.width
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = target_size[1] - source.height

    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Adjust mask generation based on alignment
    if alignment == "Middle":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y + overlap),
            (margin_x + source.width - overlap, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Left":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width - overlap, margin_y + source.height)
        ], fill=0)
    elif alignment == "Right":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)
    elif alignment == "Top":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Bottom":
        mask_draw.rectangle([
            (margin_x, margin_y + overlap),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input} , high quality, 4k"

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, DEVICE, True)

    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        yield cnet_image, image

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    yield background, cnet_image

def create_zoom_animation(previous_frame, next_frame, steps):
    # List to store all frames
    interpolated_frames = []
    
    for i in range(steps):
        t = i / (steps - 1)  # Normalized time between 0 and 1
        
        # Compute zoom factor (from 1 to 2)
        z = 1 + t  # Zoom factor increases from 1 to 2
        
        if i < steps - 1:
            # Compute crop size
            crop_size = int(1024 / z)
            
            # Compute crop coordinates to center the crop
            x0 = (1024 - crop_size) // 2
            y0 = (1024 - crop_size) // 2
            x1 = x0 + crop_size
            y1 = y0 + crop_size
            
            # Crop the previous_frame
            cropped_prev = previous_frame.crop((x0, y0, x1, y1))
            # Resize to 512x512
            resized_frame = cropped_prev.resize((512, 512), Image.LANCZOS)
            interpolated_frames.append(resized_frame)
        else:
            # For the last frame, use the next_frame resized to 512x512
            resized_frame = next_frame.resize((512, 512), Image.LANCZOS)

    return interpolated_frames

def create_video_from_images(image_list, fps=24):
    if not image_list:
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        video_path = temp_video_file.name

    frame = np.array(image_list[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in image_list:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    video.release()
    return video_path

def loop_outpainting(image, width=1024, height=1024, overlap_width=6, num_inference_steps=8, 
                     resize_option="custom", custom_resize_size=512, prompt_input=None, 
                     alignment="Middle", num_iterations=6, fps=24, num_interpolation_frames=18,
                     progress=gr.Progress()):
    current_image = image
    if(current_image.width != 1024 or current_image.height != 1024):
        for first_result in infer(current_image, 1024, 1024, overlap_width, num_inference_steps, 
                                 resize_option, 1024, prompt_input, alignment):
            pass
        current_image = first_result[1] 
        
    image_list = [current_image]
    for _ in progress.tqdm(range(num_iterations), desc="Generating frames"):
        # Generate new image
        for step_result in infer(current_image, width, height, overlap_width, num_inference_steps, 
                                 resize_option, custom_resize_size, prompt_input, alignment):
            pass  # Process all steps
        
        new_image = step_result[1]  # Get the final image from the last step
        image_list.append(new_image)
        
        # Use new image as input for next iteration
        current_image = new_image

    # Reverse the image list to create a zoom-in effect
    reverse_image_list = image_list[::-1]
    
    # Create interpolated frames
    final_frame_list = []
    
    for i in range(len(reverse_image_list) - 1):
        larger_frame = reverse_image_list[i]
        smaller_frame = reverse_image_list[i + 1]
        interpolated_frames = create_zoom_animation(larger_frame, smaller_frame, num_interpolation_frames)
        
        if i == 0:
            # Include all frames for the first sequence
            final_frame_list.extend(interpolated_frames)
        else:
            # Exclude the first frame to avoid duplication
            final_frame_list.extend(interpolated_frames[1:])

    # Create video from the final frame list
    video_path = create_video_from_images(final_frame_list, fps)
    return video_path
    
#loop_outpainting.zerogpu = True

def clear_result():
    """Clears the result ImageSlider."""
    return gr.update(value=None)

def preload_presets(target_ratio, ui_width, ui_height):
    """Updates the width and height sliders based on the selected aspect ratio."""
    if target_ratio == "9:16":
        changed_width = 720
        changed_height = 1280
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 720
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)

def select_the_right_preset(user_width, user_height):
    if user_width == 720 and user_height == 1280:
        return "9:16"
    elif user_width == 1280 and user_height == 720:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"

def toggle_custom_resize_slider(resize_option):
    return gr.update(visible=(resize_option == "Custom"))

css = """
.gradio-container {
    width: 1200px !important;
}
"""

title = """
<h1 align="center">Outpaint Video Zoom-In</h1>
<p>Ported from: <a href="https://huggingface.co/spaces/multimodalart/outpaint-video-zoom" target="_blank">outpaint-video-zoom</a> <p>
"""
with gr.Blocks(css=css) as demo:
    with gr.Column():
        gr.HTML(title)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image"
                )
                prompt_input = gr.Textbox(label="Prompt (Optional)", visible=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        run_button = gr.Button("Generate", visible=False)
                        loop_button = gr.Button("Create outpainting video")
                with gr.Row():
                    target_ratio = gr.Radio(
                        label="Expected Ratio",
                        choices=["9:16", "16:9", "1:1", "Custom"],
                        value="1:1",
                        scale=2,
                        visible=False
                    )
                    
                    alignment_dropdown = gr.Dropdown(
                        choices=["Middle", "Left", "Right", "Top", "Bottom"],
                        value="Middle",
                        label="Alignment",
                        visible=False
                    )

                with gr.Accordion(label="Advanced settings", open=False, visible=True) as settings_panel:
                    with gr.Column():
                        with gr.Row():
                            width_slider = gr.Slider(
                                label="Width",
                                minimum=720,
                                maximum=1536,
                                step=8,
                                value=1024,
                            )
                            height_slider = gr.Slider(
                                label="Height",
                                minimum=720,
                                maximum=1536,
                                step=8,
                                value=1024,
                            )
                        with gr.Row():
                            num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
                            overlap_width = gr.Slider(
                                label="Mask overlap width",
                                minimum=1,
                                maximum=50,
                                value=1,
                                step=1
                            )
                        with gr.Row():
                            resize_option = gr.Radio(
                                label="Resize input image",
                                choices=["Full", "1/2", "1/3", "1/4", "Custom"],
                                value="Custom"
                            )
                            custom_resize_size = gr.Slider(
                                label="Custom resize size",
                                minimum=64,
                                maximum=1024,
                                step=8,
                                value=512,
                                visible=False
                            )
                        with gr.Row():
                            num_iterations = gr.Slider(label="Number of iterations", minimum=2, maximum=24, step=1, value=6)
                            fps = gr.Slider(label="fps", minimum=1, maximum=24, value=24)
                        with gr.Row():
                            num_interpolation_frames = gr.Slider(label="Interpolation frames", minimum=0, maximum=10, step=1, value=18)

            with gr.Column():
                result = ImageSlider(
                    interactive=False,
                    label="Generated Image",
                    visible=False
                )
                use_as_input_button = gr.Button("Use as Input Image", visible=False)
                video_output = gr.Video(label="Outpainting Video")

    gr.Examples(
        examples=["hide.png", "disaster.png"],
        fn=loop_outpainting,
        inputs=input_image,
        outputs=video_output,
        #cache_examples="lazy",
    )

    def use_output_as_input(output_image):
        """Sets the generated output as the new input image."""
        return gr.update(value=output_image[1])

    use_as_input_button.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_image]
    )
    
    target_ratio.change(
        fn=preload_presets,
        inputs=[target_ratio, width_slider, height_slider],
        outputs=[width_slider, height_slider, settings_panel],
        queue=False
    )

    width_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    height_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    resize_option.change(
        fn=toggle_custom_resize_slider,
        inputs=[resize_option],
        outputs=[custom_resize_size],
        queue=False
    )
    
    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_width, num_inference_steps,
                resize_option, custom_resize_size, prompt_input, alignment_dropdown],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    prompt_input.submit(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_width, num_inference_steps,
                resize_option, custom_resize_size, prompt_input, alignment_dropdown],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    loop_button.click(
        fn=loop_outpainting,
        inputs=[input_image, width_slider, height_slider, overlap_width, num_inference_steps,
                resize_option, custom_resize_size, prompt_input, alignment_dropdown, 
                num_iterations, fps, num_interpolation_frames],
        outputs=video_output,
    )

demo.launch(share=False)