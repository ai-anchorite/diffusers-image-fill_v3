module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {},
        message: [
          "python {{args.type === 'outpaint' ? 'app-outpaint.py' : args.type === 'zoom' ? 'app-zoom.py' : 'app.py'}}",
        ],
        on: [{
          event: "/http:\\/\\/[a-zA-Z0-9.:]+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
