const path = require('path')
module.exports = {
  version: "2.0",
  title: "diffusers-image-fill",
  description: "Inpaint or Remove objects from an image - https://huggingface.co/spaces/OzzyGT/diffusers-image-fill - or Outpaint - https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint",
  icon: "icon.gif",
  menu: async (kernel, info) => {
    let installed = info.exists("env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      start2: info.running("start-outpaint.js"),
      start3: info.running("start-zoom.js"),
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
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.start2) {
        let local = info.local("start-outpaint.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start-outpaint.js",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start-outpaint.js",
          }]
        }
      } else if (running.start3) {
        let local = info.local("start-zoom.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start-zoom.js",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start-zoom.js",
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
          text: "Deleting Gradio Cache",
          href: "delete-cache.js",
          confirm:"Are you sure you wish to delete the Gradio cache?"
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start Inpaint/Remove",
          href: "start.js",
        }, {
          icon: "fa-solid fa-power-off",
          text: "Start Outpaint",
          href: "start-outpaint.js",
        }, {
          icon: "fa-solid fa-power-off",
          text: "Start Outpaint Zooooom",
          href: "start-zoom.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Delete Gradio Cache",
          href: "delete-cache.js",
          confirm:"Are you sure you wish to delete the Gradio cache?"
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
