import gradio as gr
import torch
import devicetorch
import os
import webbrowser

from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
}
DEVICE = devicetorch.get(torch)
MAX_GALLERY_IMAGES = 20 
OUTPUT_DIR = "outputs" 

pipe = None
global_image = None
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


def fill_image(prompt, image, model_selection, guidance_scale, steps, paste_back, auto_save):
    global latest_result, gallery_images 
    init()
    source = image["background"]
    mask = image["layers"][0]

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


    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        image=cnet_image,
    ):

        yield (image, cnet_image), None

    if paste_back:
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), binary_mask)
        latest_result = cnet_image
    else:
        latest_result = image.convert("RGBA")

    # Use original save_output function
    save_path, filename = save_output(latest_result, auto_save)
    
    # Add to gallery
    gallery_images.append((latest_result, filename))
    gallery_update = [(img, fname) for img, fname in gallery_images]

    # Final yield with the completed image
    yield (source, latest_result), gallery_update

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
        # Ensure the outputs directory exists
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
        _, filename = list(gallery_images)[evt.index]
        selected_gallery_image = filename
        return f"Selected image: {filename}", gr.update(visible=True)
    return "Invalid selection", gr.update(visible=False)


def set_img(image):
    global global_image
    global_image = image["background"]

def resize(image, size):
    global global_image
    size = (int(size) // 8) * 8
    source = global_image.copy()
    source.thumbnail((size, size), Image.LANCZOS)

    resized_w, resized_h = source.size
    resized_w = (resized_w // 8) * 8
    resized_h = (resized_h // 8) * 8
    source = source.crop((0, 0, resized_w, resized_h))

    w, h = global_image.size
    max = (w // 8) * 8
    return gr.update(value=source, canvas_size=(w,h)), gr.update(maximum=max, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

# send result back to input
def send_to_input(result_slider):
    if latest_result is not None and result_slider is not None:
        global global_image
        global_image = latest_result
        return gr.update(value=latest_result), gr.update(value=None)
    # If no result or result slider is empty, don't update anything
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
        )
        
    with gr.Row():
        run_button = gr.Button("Generate", scale=3)
        paste_back = gr.Checkbox(True, label="Paste back original", scale=1)
        auto_save = gr.Checkbox(True, label="Autosave all results", scale=1)
        
    with gr.Row():
        model_selection = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="RealVisXL V5.0 Lightning",
            label="Model",
        )
        prompt = gr.Textbox(value="high quality, 4K", label="Prompt (for adding details via inpaint)", visible=True)
        size = gr.Slider(value=1024, label="Resize", minimum=0, maximum=1024, step=8, visible=True, interactive=True)
        guidance_scale = gr.Number(value=1.5, label="Guidance Scale", minimum=1.5, maximum=8, step=0.5, visible=True)
        steps = gr.Number(value=8, label="Steps", precision=0, visible=True)
        
    with gr.Row():   
        open_folder_button = gr.Button("Open Outputs Folder")
        console_info = gr.Textbox(label="Console", interactive=False) 
        result_to_input = gr.Button("Send Result to Input")  
        
    with gr.Row():
        gallery = gr.Gallery(
            label=f"Image Gallery (most recent {MAX_GALLERY_IMAGES} images)",
            show_label=True,
            elem_id="gallery",
            preview=True,
            object_fit="contain",
            columns=5,
            height=400,
            show_download_button=False
        )
        
    with gr.Row():    
        clear_gallery_btn = gr.Button("Clear Gallery", scale=1)
        save_selected_btn = gr.Button("Save Selected", visible=True)
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
        inputs=[prompt, input_image, model_selection, guidance_scale, steps, paste_back, auto_save],
        outputs=[result, gallery]
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
    
    open_folder_button.click(
        fn=open_outputs_folder,
        inputs=None,
        outputs=console_info
    )

    input_image.upload(
        fn=set_img,
        inputs=input_image
    ).then(
        fn=resize, 
        inputs=[input_image, size], 
        outputs=[input_image, size, guidance_scale, steps, prompt]
    )
    
    size.change(
        fn=resize, 
        inputs=[input_image, size], 
        outputs=[input_image, size, guidance_scale, steps, prompt]
    )


demo.launch(share=False)
