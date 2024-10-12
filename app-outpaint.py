import gradio as gr
import torch
import os
import numpy as np
import webbrowser
import time

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
import devicetorch

from gradio import SelectData
from datetime import datetime
from pathlib import Path
from collections import deque
from PIL import Image, ImageDraw

MAX_GALLERY_IMAGES = 20
latest_result = None
gallery_images = deque(maxlen=MAX_GALLERY_IMAGES)
selected_image_index = None



DEVICE = devicetorch.get(torch)

OUTPUT_DIR = "outputs"


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

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height


    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, preview_state):
    if preview_state == "Off" or image is None:
        return gr.update(visible=False), gr.update(value="Off")
    
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    # Create a preview image showing the mask
    preview = background.copy().convert('RGBA')
    
    # Create a semi-transparent red overlay
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))  # Reduced alpha to 64 (25% opacity)
    
    # Convert black pixels in the mask to semi-transparent red
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask)
    
    # Overlay the red mask on the background
    preview = Image.alpha_composite(preview, red_mask)
    
    return gr.update(value=preview, visible=True), gr.update(value="On")

def infer(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"

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

def clear_result():
    """Clears the result ImageSlider."""
    return gr.update(value=None)

def preload_presets(target_ratio, ui_width, ui_height):
    """Updates the width and height sliders based on the selected aspect ratio."""
    if target_ratio == "9:16":
        changed_width = 720
        changed_height = 1280
        return changed_width, changed_height, gr.update()
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 720
        return changed_width, changed_height, gr.update()
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update()
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

def update_history(new_image, auto_save_checked):
    global gallery_images
    if new_image is not None:
        # Extract the image from the tuple if necessary
        if isinstance(new_image, tuple):
            new_image = new_image[1] if len(new_image) > 1 else new_image[0]
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(new_image, np.ndarray):
            new_image = Image.fromarray(new_image.astype('uint8'))
        elif isinstance(new_image, str):
            new_image = Image.open(new_image)
        
        filename = f"outp_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        gallery_images.appendleft((new_image, filename))
        if auto_save_checked:
            save_output(new_image, True, filename)
    return gr.update(value=list(gallery_images))


def save_output(latest_result, auto_save, filename):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if auto_save:
            if isinstance(latest_result, np.ndarray):
                latest_result = Image.fromarray(latest_result.astype('uint8'))
            latest_result.save(full_path)
            print(f"Image auto-saved as: {full_path}")
            return full_path, filename
        else:
            print(f"Auto-save disabled, assigned filename: {filename}")
            return None, filename
    except Exception as e:
        print(f"Error handling image path/save: {e}")
        return None, None

def update_selected_image(evt: gr.SelectData, gallery_state):
    global selected_image_index
    selected_image_index = evt.index
    if evt.index < len(gallery_state):
        _, filename = gallery_state[evt.index]
        return f"Selected image: {filename}"
    return "Invalid selection"
    
def save_selected_image(gallery_state):
    global selected_image_index
    if selected_image_index is None or gallery_state is None:
        return "Please select an image first"
    
    try:
        selected_image, filename = gallery_state[selected_image_index]
        
        print(f"Selected image index: {selected_image_index}")
        print(f"Filename: {filename}")
        
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return f"Image already saved as: {filename}"
        
        if isinstance(selected_image, Image.Image):
            selected_image.save(full_path)
        elif isinstance(selected_image, np.ndarray):
            Image.fromarray(selected_image.astype('uint8')).save(full_path)
        elif isinstance(selected_image, str) and os.path.isfile(selected_image):
            import shutil
            shutil.copy(selected_image, full_path)
        else:
            return f"Unsupported image type: {type(selected_image)}"
        
        print(f"Image saved as: {filename}")
        return f"Image saved as: {filename}"
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Error saving image: {str(e)}"
        
def use_output_as_input(output_image):
    """Sets the generated output as the new input image."""
    return gr.update(value=output_image[1])    

def open_outputs_folder():
    try:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        folder_uri = Path(OUTPUT_DIR).absolute().as_uri()
        webbrowser.open(folder_uri)
        return "Opened outputs folder (folder can be shy and hide behind active windows)."
    except Exception as e:
        return f"Error opening outputs folder: {str(e)}"
        
    
# def clear_gallery():
    # global gallery_images, selected_gallery_image
    # gallery_images.clear()
    # selected_gallery_image = None
    # return gr.update(value=None), gr.update(visible=False), gr.update(value=None)

    
css = """
.gradio-container {
    width: 1200px !important;
}
"""

title = """<h1 align="center">Diffusers Image Outpaint</h1>
<div align="center">Drop an image you would like to extend, pick your expected ratio and hit Generate.</div>
"""

#THE UI      

with gr.Blocks(fill_width=True) as demo:
    gr.HTML(title)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        result = ImageSlider(interactive=False, label="Generated Image")

    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt (Optional)", scale=4)
        auto_save = gr.Checkbox(label="Auto-save", value=True, scale=1)
        run_button = gr.Button("Generate", variant="primary", scale=1)
        use_as_input_button = gr.Button("Use as Input Image", visible=True, scale=1)

    with gr.Row():
        target_ratio = gr.Radio(
            label="Expected Ratio",
            choices=["9:16", "16:9", "1:1", "Custom"],
            value="9:16",
        )
        
        #preview_button = gr.Button("Preview alignment and mask")
        preview_toggle = gr.Radio(
            choices=["Off", "On"],
            value="Off",
            label="Preview Window",
            interactive=True
        )
         
        alignment_dropdown = gr.Dropdown(
            choices=["Middle", "Left", "Right", "Top", "Bottom"],
            value="Middle",
            label="Alignment"
        )
        resize_option = gr.Radio(
            label="Resize input image",
            choices=["Full", "50%", "33%", "25%", "Custom"],
            value="Full"
        )

    with gr.Accordion(label="Advanced settings", open=False):
        with gr.Row():
            width_slider = gr.Slider(
                label="Target Width",
                minimum=720,
                maximum=1536,
                step=8,
                value=720,
            )
            height_slider = gr.Slider(
                label="Target Height",
                minimum=720,
                maximum=1536,
                step=8,
                value=1280,
            )
        
        with gr.Row():
            num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
            overlap_percentage = gr.Slider(
                label="Mask overlap (%)",
                minimum=1,
                maximum=50,
                value=10,
                step=1
            )

        with gr.Row():
            overlap_top = gr.Checkbox(label="Overlap Top", value=True)
            overlap_right = gr.Checkbox(label="Overlap Right", value=True)
            overlap_left = gr.Checkbox(label="Overlap Left", value=True)
            overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)

        custom_resize_percentage = gr.Slider(
            label="Custom resize (%)",
            minimum=1,
            maximum=100,
            step=1,
            value=50,
            visible=False
        )
        
        #preview_button = gr.Button("Preview alignment and mask")

    gallery = gr.Gallery(
        label=f"Image Gallery (most recent {MAX_GALLERY_IMAGES} images)",
        show_label=True,
        elem_id="gallery",
        columns=5,
        height="auto",
        object_fit="contain",
        allow_preview=True,
        preview=False,
    )

    with gr.Row():
        save_selected_btn = gr.Button("Save Selected")
        gallery_status = gr.Textbox(label="Gallery Status", interactive=False)
        open_folder_button = gr.Button("Open Outputs Folder", variant="secondary") 

    preview_image = gr.Image(label="Preview", visible=False)
    

    
    preview_toggle.change(
        fn=preview_image_and_mask,
        inputs=[
            input_image, width_slider, height_slider, overlap_percentage, 
            resize_option, custom_resize_percentage, alignment_dropdown,
            overlap_left, overlap_right, overlap_top, overlap_bottom,
            preview_toggle
        ],
        outputs=[preview_image, preview_toggle]
    ) 

    
    def toggle_preview_visibility(choice):
        return gr.update(visible=(choice == "On"))

    preview_toggle.change(
        fn=toggle_preview_visibility,
        inputs=[preview_toggle],
        outputs=[preview_image]
    )
    
    use_as_input_button.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_image]
    )
    
    target_ratio.change(
        fn=preload_presets,
        inputs=[target_ratio, width_slider, height_slider],
        outputs=[width_slider, height_slider],
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
        outputs=[custom_resize_percentage],
        queue=False
    )
    
    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=result,
    ).then(
        fn=update_history,
        inputs=[result, auto_save],
        outputs=[gallery],
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
        inputs=[input_image, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=result,
    ).then(
        fn=update_history,
        inputs=[result, auto_save],
        outputs=[gallery],
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )


    gallery.select(
        fn=update_selected_image,
        inputs=[gallery],
        outputs=[gallery_status]
    )

    save_selected_btn.click(
        fn=save_selected_image,
        inputs=[gallery],
        outputs=[gallery_status]
    )   
    
    open_folder_button.click(
        fn=open_outputs_folder,
        inputs=None,
        outputs=gallery_status
    )

demo.launch(share=False)