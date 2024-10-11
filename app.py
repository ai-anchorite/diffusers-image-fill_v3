import gradio as gr
import torch
import devicetorch
import os
import webbrowser
import math

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from huggingface_hub import hf_hub_download
from gradio_imageslider import ImageSlider
from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image

DEVICE = devicetorch.get(torch)

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
}


MAX_GALLERY_IMAGES = 20 
MIN_IMAGE_SIZE = 512 
OUTPUT_DIR = "outputs" 
VAE_SCALE_FACTOR = 8

last_resize_time = 0
RESIZE_COOLDOWN = 0.5  # Seconds

pipe = None
global_image = None
global_original_image = None 
latest_result = None
gallery_images = deque(maxlen=MAX_GALLERY_IMAGES)
selected_gallery_image = None

def init():
    global pipe

    if pipe is None:
        
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


def fill_image(prompt, image, model_selection, guidance_scale, steps, paste_back, auto_save, num_images):
    global latest_result, gallery_images, global_image
    init()
    source = global_image  # Use the current global_image instead of image["background"]
    mask = image["layers"][0]

    # Ensure source image meets size requirements
    source = resize_image(source, min_size=MIN_IMAGE_SIZE, scale_factor=VAE_SCALE_FACTOR)
    
    # Resize mask to match source image size
    mask = mask.resize(source.size, Image.LANCZOS)
    
    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, DEVICE, True)

    all_results = []
    
    for n in range(num_images):
        intermediate_images = []
        
        # Use source image when updating console
        f"Starting generation of image {n+1} of {num_images}..."
        
        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            image=cnet_image,
        ):
            yield (image, cnet_image), gr.update(value=None, interactive=False), f"Generating image {n+1} of {num_images}..."
            intermediate_images.append(image)

        final_image = intermediate_images[-1]
        if paste_back:
            final_image = final_image.convert("RGBA")
            result_image = cnet_image.copy()
            result_image.paste(final_image, (0, 0), binary_mask)
        else:
            result_image = final_image.convert("RGBA")
        
        all_results.append(result_image)
        
        save_path, filename = save_output(result_image, auto_save)
        gallery_images.append((result_image, filename))
        
      
        gallery_update = [(img, fname) for img, fname in gallery_images]
        yield (source, result_image), gallery_update, f"Completed image {n+1} of {num_images}"

    latest_result = all_results[-1]
    yield (source, latest_result), gallery_update, "All images generated successfully!"


def save_output(latest_result, auto_save): 
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        new_filename = f"img_{current_time}.png"
        full_path = os.path.join(OUTPUT_DIR, new_filename)
        
        if auto_save:
            latest_result.save(full_path)
            print(f"Image auto-saved as: {full_path}")
            return full_path, new_filename
        else:
            print(f"Auto-save disabled, assigned filename: {new_filename}")
            return None, new_filename
    except Exception as e:
        print(f"Error handling image path/save: {e}")
        return None, None


def save_selected_image():
    global selected_gallery_image
    if selected_gallery_image is None:
        return "Please select an image first"
        
    for image, filename in gallery_images:
        if filename == selected_gallery_image:
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                full_path = os.path.join(OUTPUT_DIR, filename)
                image.save(full_path)
                return f"Image saved as: {filename}"
            except Exception as e:
                return f"Error saving image: {str(e)}"
    return "Selected image no longer in gallery"


def clear_gallery():
    global gallery_images, selected_gallery_image
    gallery_images.clear()
    selected_gallery_image = None
    return gr.update(value=None), gr.update(visible=False), gr.update(value=None)
  
  
def open_outputs_folder():
    try:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        folder_uri = Path(OUTPUT_DIR).absolute().as_uri()
        webbrowser.open(folder_uri)
        return "Opened outputs folder (folder can be shy and hide behind active windows)."
    except Exception as e:
        return f"Error opening outputs folder: {str(e)}"
    
    
def clear_result():
    return gr.update(value=None)


def select_gallery_image(evt: gr.SelectData):
    global selected_gallery_image
    if evt.index < len(gallery_images):
        selected_image, filename = list(gallery_images)[evt.index]
        selected_gallery_image = selected_image
        return f"Selected image: {filename}", gr.update(visible=True)
    return "Invalid selection", gr.update(visible=False)


def send_selected_to_input():
    global selected_gallery_image, global_image, global_original_image
    try:
        if selected_gallery_image is not None:
            global_image = selected_gallery_image.copy()
            global_original_image = selected_gallery_image.copy()
            
            # Update resize controls for the new image
            resize_slider_update, resize_button_update, console_info_update = update_resize_controls({"background": global_image})
            
            # Always set to 100% if resize options are available
            if isinstance(resize_slider_update, dict) and 'choices' in resize_slider_update and resize_slider_update['choices']:
                resize_slider_update = gr.update(value=100, choices=resize_slider_update['choices'], interactive=True)
            else:
                resize_slider_update = gr.update(value=None, choices=[], interactive=False)
            
            return (
                gr.update(value=global_image),  # Update input image
                resize_slider_update,  # Update resize slider options and value
                resize_button_update,  # Update resize button state
                console_info_update,  # Update console info
                "Selected image successfully sent to input"  # Status message
            )
        else:
            return gr.update(), gr.update(value=None, choices=[], interactive=False), gr.update(interactive=False), gr.update(), "No image selected. Please select an image from the gallery first."
    except Exception as e:
        print(f"Error sending image to input: {str(e)}")
        return gr.update(), gr.update(value=None, choices=[], interactive=False), gr.update(interactive=False), gr.update(), f"Error sending image to input: {str(e)}"
    
    
def set_img(image, size_slider):
    global global_image
    if image is None or image["background"] is None:
        return gr.update(value="No image loaded"), gr.update()
    
    global_image = image["background"]
    min_percentage = calculate_min_resize_percentage(global_image)
    
    return (
        update_image_info(size_slider),
        gr.update(minimum=min_percentage)
    )
    
def resize_image(image, min_size=MIN_IMAGE_SIZE, scale_factor=VAE_SCALE_FACTOR):
    width, height = image.size
    
    # Ensure minimum size
    if width < min_size or height < min_size:
        scale = max(min_size / width, min_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height
    
    # Round to nearest multiple of scale_factor
    new_width = ((new_width + scale_factor - 1) // scale_factor) * scale_factor
    new_height = ((new_height + scale_factor - 1) // scale_factor) * scale_factor
    
    return image.resize((new_width, new_height), Image.LANCZOS)
    
    
def preview_resize(percentage):
    global global_image, global_original_image
    if global_original_image is None:
        return "No image loaded. Please upload an image first."
    if percentage is None:
        return "No resize options available. The image is already at the minimum allowed size."
    
    original_w, original_h = global_original_image.size
    current_w, current_h = global_image.size
    new_w = ((original_w * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    new_h = ((original_h * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    
    if new_w < MIN_IMAGE_SIZE or new_h < MIN_IMAGE_SIZE:
        return f"Cannot resize below minimum allowed size of {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}."
    
    mp = (new_w * new_h) / 1000000
    estimated_vram = 10.0 + (mp * 4) + 1.0
    
    if percentage == 100:
        resize_info = '<p style="color: #4CAF50;">Image is at 100% of its original size. No resize needed.</p>'
    elif new_w == current_w and new_h == current_h:
        resize_info = '<p style="color: #FF6347;">Preview size = current size. No resize required.</p>'
    else:
        resize_info = '<p style="color: #4CAF50;">Click \'Apply Resize\' to perform this resize.</p>'
    
    return f"""
<p><span style="color: #FF8C00; font-weight: bold;">Preview for {percentage}% of original size:</span></p>
<p>New size would be: <span style="color: #3498DB;">{new_w}x{new_h}</span> (current size: <span style="color: #3498DB;">{current_w}x{current_h}</span>)</p>
<p>Estimated VRAM usage: <span style="color: #9932CC;">{estimated_vram:.1f}GB</span></p>
{resize_info}
"""


def calculate_resize_options(image):
    if image is None:
        return []
    
    original_w, original_h = image.size
    aspect_ratio = original_w / original_h
    
    # Define a range of percentages to check
    percentages = list(range(100, 9, -10))  # 100%, 90%, 80%, ..., 10%
    
    valid_options = []
    min_dimension = min(original_w, original_h)
    
    for percentage in percentages:
        new_w = int((original_w * percentage) / 100)
        new_h = int((original_h * percentage) / 100)
        
        # Ensure dimensions are multiples of VAE_SCALE_FACTOR
        new_w = (new_w // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
        new_h = (new_h // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
        
        # Check if resized dimensions meet minimum requirements
        if new_w >= MIN_IMAGE_SIZE and new_h >= MIN_IMAGE_SIZE:
            valid_options.append(percentage)
        else:
            # If we've hit the minimum size, calculate the exact percentage needed
            if not valid_options:
                min_percentage = max((MIN_IMAGE_SIZE / original_w) * 100, (MIN_IMAGE_SIZE / original_h) * 100)
                min_percentage = math.ceil(min_percentage)
                valid_options.append(min_percentage)
            break
    
    # Remove 100% if it's the only option
    if len(valid_options) == 1 and valid_options[0] == 100:
        return []
    
    # Ensure we have at least 3 options (if possible) by adding intermediate values
    while len(valid_options) < 3 and len(valid_options) < len(percentages):
        for i in range(len(valid_options) - 1):
            mid_percentage = (valid_options[i] + valid_options[i+1]) // 2
            if mid_percentage not in valid_options:
                valid_options.insert(i+1, mid_percentage)
                if len(valid_options) == 3:
                    break
    
    return sorted(valid_options, reverse=True)

def update_resize_controls(image):
    global global_image, global_original_image
    if image is None or image["background"] is None:
        return (
            gr.update(value=None, choices=[], interactive=False),
            gr.update(interactive=False),
            "No image loaded. Please upload an image."
        )
    
    global_original_image = image["background"].copy()
    global_image = global_original_image.copy()
    
    width, height = global_original_image.size
    mp = (width * height) / 1000000
    estimated_vram = 10.0 + (mp * 4) + 1.0
    
    resize_options = calculate_resize_options(global_original_image)
    
    if not resize_options:
        info_text = f"""
<p><span style="color: #FF8C00; font-weight: bold;">Current size:</span> {width}x{height}</p>
<p><span style="color: #FF6347;">No resize options available. Image is at the minimum allowed size.</span></p>
<p><span style="color: #9932CC;">Estimated VRAM usage: {estimated_vram:.1f}GB</span></p>
<p><span style="color: #1E90FF;">Minimum allowed size: {MIN_IMAGE_SIZE} pixels for the smallest dimension</p>
"""
        dropdown_update = gr.update(value=None, choices=[], interactive=False)
    else:
        info_text = f"""
<p><span style="color: #FF8C00; font-weight: bold;">Current size:</span> {width}x{height}</p>
<p><span style="color: #4CAF50;">Available resize options: {", ".join(f"{opt}%" for opt in resize_options)}</span></p>
<p><span style="color: #9932CC;">Estimated VRAM usage: {estimated_vram:.1f}GB</span></p>
<p><span style="color: #1E90FF;">Minimum allowed size: {MIN_IMAGE_SIZE} pixels for the smallest dimension</p>
"""
        dropdown_update = gr.update(value=resize_options[0], choices=resize_options, interactive=True)
    
    return (
        dropdown_update,
        gr.update(interactive=bool(resize_options)),
        info_text
    )

def apply_resize(percentage):
    global global_image, global_original_image
    if global_original_image is None:
        return None, "No image loaded"
    
    original_w, original_h = global_original_image.size
    new_w = ((original_w * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    new_h = ((original_h * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    
    if percentage == 100:
        global_image = global_original_image.copy()
        return global_image, f"Image restored to original copy: <span style=\"color: #3498DB;\">{original_w}x{original_h}</span>"
    else:
        global_image = global_original_image.resize((new_w, new_h), Image.LANCZOS)
        return global_image, f"Image resized to <span style=\"color: #3498DB;\">{new_w}x{new_h}</span>"


def send_to_input(result_slider):
    if latest_result is not None and result_slider is not None:
        global global_image
        global_image = latest_result
        return gr.update(value=latest_result), gr.update(value=None)
    return gr.update(), gr.update()
    

with gr.Blocks(fill_width=True) as demo:
    with gr.Row():
        input_image = gr.ImageMask(
            type="pil",
            label="Input Image",
            layers=False,
            sources=["upload"],
        )
        result = ImageSlider(
            interactive=False,
            label="Generated Image",
            elem_classes=["image-slider-custom"],
        )
        
    with gr.Row(variant = "compact"):
        run_button = gr.Button("Generate", scale=3)
        num_images = gr.Slider(value=1, label="Number of Images", minimum=1,maximum=10, step=1, scale=2)
        paste_back = gr.Checkbox(True, label="Paste back original background", scale=1)
        auto_save = gr.Checkbox(True, label="Autosave all results", scale=1)
        unload_model = gr.Button("Unload all (placeholder work-in-progress)", variant="primary", size = "sm")
        
    with gr.Row():
        model_selection = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="RealVisXL V5.0 Lightning",
            label="Model",
        )
        prompt = gr.Textbox(value="high quality, 4K", label="Prompt (for adding details via inpaint)", visible=True)

        resize_slider = gr.Dropdown(
            choices=[],
            value=None,
            label="Resize Options (%)",
            interactive=False
        )
        resize_button = gr.Button("Apply Resize", interactive=False)
        guidance_scale = gr.Number(value=1.5, label="Guidance Scale", minimum=1.5, maximum=8, step=0.5, visible=True)
        steps = gr.Number(value=8, label="Steps", precision=0, visible=True)
        
    with gr.Row(variant="compact"):   
        open_folder_button = gr.Button("Open Outputs Folder", variant="secondary")
        console_info = gr.HTML(label="Console")
        result_to_input = gr.Button("Send Result to Input", variant="secondary")
        
    with gr.Row():
        gallery = gr.Gallery(
            label=f"Image Gallery (most recent {MAX_GALLERY_IMAGES} images)",
            show_label=True,
            elem_id="gallery",
            preview=False,  # Start with preview disabled
            object_fit="contain",
            columns=5,
            height=400,
            show_download_button=False
        )
        
    with gr.Row():    
        clear_gallery_btn = gr.Button("Clear Gallery", scale=1)
        save_selected_btn = gr.Button("Save Selected", visible=True)
        send_gallery_to_input_btn = gr.Button("Send Selected to Input", visible=True)
        gallery_status = gr.Textbox(label="Gallery Status", interactive=False)
   
   
    # event handlers        
    run_button.click(
        fn=lambda: "Preparing Image Generation...", 
        inputs=None,
        outputs=console_info,
    ).then(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, guidance_scale, steps, paste_back, auto_save, num_images],
        outputs=[result, gallery, console_info]
    )

    gallery.select(
        fn=select_gallery_image,
        outputs=[gallery_status, save_selected_btn]
    )
    
    save_selected_btn.click(
        fn=save_selected_image,
        outputs=gallery_status
    )
    
    clear_gallery_btn.click(
        fn=clear_gallery,
        outputs=[gallery, save_selected_btn, gallery_status]
    )
    
    result_to_input.click(
        fn=send_to_input,
        inputs=[result], 
        outputs=[input_image, result],
    )
    
    send_gallery_to_input_btn.click(
        fn=send_selected_to_input,
        outputs=[input_image, resize_slider, resize_button, console_info, gallery_status]
    )
    
    open_folder_button.click(
        fn=open_outputs_folder,
        inputs=None,
        outputs=console_info
    )

    input_image.upload(
        fn=update_resize_controls,
        inputs=[input_image],
        outputs=[resize_slider, resize_button, console_info]
    )

    resize_slider.change(
        fn=preview_resize,
        inputs=[resize_slider],
        outputs=[console_info]
    )

    resize_button.click(
        fn=apply_resize,
        inputs=[resize_slider],
        outputs=[input_image, console_info]
    )
 
demo.launch(share=False)