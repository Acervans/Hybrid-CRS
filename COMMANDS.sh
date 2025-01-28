docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama


# Backend API Tunneling

# https://dashboard.ngrok.com/get-started/setup/linux
ngrok http --url=https://perfectly-large-goshawk.ngrok-free.app 3000

# https://gitlab.com/pyjam.as/tunnel
curl https://tunnel.pyjam.as/{PORT} > tunnel.conf && wg-quick up ./tunnel.conf
wg-quick up ./tunnel.conf
wg-quick down ./tunnel.conf

# Other alternatives:
    # https://theboroer.github.io/localtunnel-www/ /usr/bin/lt --port {PORT}
    # Cloudflare Tunnels
