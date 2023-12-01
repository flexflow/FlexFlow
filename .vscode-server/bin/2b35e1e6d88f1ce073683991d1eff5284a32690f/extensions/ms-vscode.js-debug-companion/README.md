# js-debug-companion

A companion extension to [js-debug](https://github.com/microsoft/vscode-js-debug) to enable remote Chrome debugging. You probably don't want to install this extension by itself, but for your interest, this is what it does.

The scenario is if you are developing in a remote environment—like WSL, a container, ssh, or [VS Codespaces](https://visualstudio.microsoft.com/services/visual-studio-codespaces/)—and are port-forwarding a server to develop (and debug) in a browser locally. For remote development, VS Code runs two sets of extensions: one on the remote machine, and one on your local computer. `js-debug` is a "workspace" extension that runs on the remote machine, but we need to launch and talk to Chrome locally.

That's where this companion extension comes in. This helper extension runs on the local machine (in the "UI") and registers a command that `js-debug` can call to launch a server. `js-debug` requests a port to be forwarded for debug traffic, and once launching a browser the companion will connect to and forward traffic over that socket.
