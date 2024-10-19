# Diffusers Image Fill 
a Pinokio install script:   [pinokio.computer](https://pinokio.computer/) <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" align="right"></a>
<table>
  <tr>
    <td width="25%" style="vertical-align: top; padding-right: 20px;">
      <img src="https://github.com/user-attachments/assets/1122b540-2ac2-4460-a4c7-c1035652cd19" width="100%"/>
    </td>
    <td width="75%" style="vertical-align: top; padding-top: 0;">
      <h2 style="margin-top: 0; margin-bottom: 20px; font-size: 24px;">🚀 3 Apps in 1!</h2>
    <p style="margin-top: 0; margin-bottom: 15px;">
  Three diffusers apps combining the speed of SDXL lightning (RealViz5) with the precision of ControlNetPlus Promax
  </p>
  
  <p style="margin-bottom: 20px;">
  Based on a Proof of Concept by OzzyGT: <a href="https://huggingface.co/blog/OzzyGT/diffusers-image-fill">https://huggingface.co/blog/OzzyGT/diffusers-image-fill</a>
  </p>
  
  <ul style="list-style-type: none; padding-left: 0; margin-top: 20px;">
    <li style="margin-bottom: 10px;">📸 remove or inpaint: <a href="https://huggingface.co/spaces/OzzyGT/diffusers-image-fill">/spaces/OzzyGT/diffusers-image-fill</a></li>
    <li style="margin-bottom: 10px;">🖌️ outpaint: <a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint">/spaces/fffiloni/diffusers-image-outpaint</a></li>
    <li style="margin-bottom: 10px;">🤖 zoom: <a href="https://huggingface.co/spaces/multimodalart/outpaint-video-zoom">/spaces/multimodalart/outpaint-video-zoom</a></li>
  </ul>
</td>
  </tr>
</table>



 Custom Gradio UIs designed especially for Pinokio
<table>
  <tr>
    <td width="50%" style="padding: 10px;">
      <img src="https://github.com/user-attachments/assets/3d6946b8-1f00-4528-937f-b03434920876" alt="DIF-inpaint" width="100%" />
      <p align="center"><em>DIF-inpaint interface</em></p>
    </td>
    <td width="50%" style="padding: 10px;">
      <img src="https://github.com/user-attachments/assets/74dbdd98-d08f-4ab0-a8ff-79cb3af3569c" alt="DIF-outpaint" width="100%" />
      <p align="center"><em>DIF-outpaint interface</em></p>
    </td>
  </tr>
</table>


## 🖥️ System Requirements & Important Notes

### Hardware & Storage:
- 32GB RAM for comfortable operation
- 16GB VRAM recommended (8GB miiight work, but performance may vary). It's snappy on a 3090.
- ~20GB free space for download. It downloads the required models on 1st generation (models are shared across apps)


### For Existing Users

If you already have Diffusers Image Fill installed in Pinokio:

1. Copy the following files from this repo to your `.\api\diffusers-image-fill.git` folder:
   - `pinokio.js`
   - `start.js`
   - `delete-cache.js`
   - `app.py`
   - `app-outpaint.py`
   - `app.zoom.py`
     
 2. These files will overwrite existing ones. No additional installation required. 

- **App Switching:** Click `Refresh` on the Pinokio top bar when switching between companion apps to update the Gradio webUI.
- **Cache Management:** Use the provided `Delete Gradio Image Cache` button periodically to manage temporary images in the Gradio cache.

- **Project Status:** This is an enhanced proof-of-concept that does what it says on the tin, with room for future improvements. Read the linked guide by <a href="https://huggingface.co/blog/OzzyGT/diffusers-image-fill">OzzyGT</a> for an idea about what's going on under the hood. I prefer dash and interior!

i was planning on combining them into one UI, but gave me a headache.. and video models are dropping left and right!
* Yes, a model loader would be cool. I tested and didn't really notice an improvement from the half dozen popular models i tried. Also, it's geared for diffusers multi file format and uses a custom SDXLpipeline that doesn't seem to work nicely with `from_single_file`. I also wasn't able to optimize any more than adding in garbage collection and cuda cache. Moving the vae to CPU was less impactful than i'd hoped!
* Adding FastSAM for inpaint masking would be cool as well. and flux.. 
* Imo, this is a fantastic app for prepping images for a Flux upscale. It's also great at generative fill compared to other alternatives. I was close to adding an upscaler using the Promax ControlNet, but I can't afford to spend anymore time on it! 

* oh, and it's mostly written by ClaudeAI ;) long as you know what you want, you can eventually do it. just have to watch for its love for creating new functions to fix broken functions.. and next minute you have half a dozen functions related to resizing an input image haha! it's effective but messy. i could spend ages refactoring, but it works well enough :)
* Enjoy!

<sub>
This project is licensed under the Apache License 2.0. It builds upon Hugging Face Spaces by OzzyGT, fffiloni, and multimodalart, also under Apache 2.0.
</sub>
