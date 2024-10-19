const path = require('path')
module.exports = {
  version: "2.0",
  title: "diffusers-image-fill",
  description: "Inpaint or Remove objects from an image - https://huggingface.co/spaces/OzzyGT/diffusers-image-fill - or Outpaint - https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint - or Outpaint Video Zoom - https://huggingface.co/spaces/multimodalart/outpaint-video-zoom",
  icon: "icon.gif",
  menu: async (kernel, info) => {
    let installed = info.exists(__dirname, "cache")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("delete-cache.js")
    }
    
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }, {
            icon: "fa-regular fa-circle-xmark",
            text: "Delete Gradio Image Cache",
            href: "delete-cache.js",
            confirm: "Are you sure you wish to delete the Gradio Image cache?"
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Deleting Gradio Image Cache",
          href: "delete-cache.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start Inpaint/Remove",
          href: "start.js",
          params: {
            type: "inpaint"
          }
        }, {
          icon: "fa-solid fa-power-off",
          text: "Start Outpaint",
          href: "start.js",
          params: {
            type: "outpaint"
          }
        }, {
          icon: "fa-solid fa-power-off",
          text: "Start Outpaint Video Zoom",
          href: "start.js",
          params: {
            type: "zoom"
          }
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
          confirm: "Are you sure you wish to update this app?"
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Delete Gradio Image Cache",
          href: "delete-cache.js",
          confirm: "Are you sure you wish to delete the Gradio Image cache?"
        }, {
          icon: 'fa-solid fa-folder',
          text: "Output Folder",
          href: "outputs?fs",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
