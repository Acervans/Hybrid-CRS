docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

npx shadcn@latest add -a -y -o

### Backend API Tunneling ###

# 1. Enable zrok environment
zrok enable <your_account_token>

# 2. Reserve unique share token
zrok reserve public 8000 --unique-name "hybridcrs"

# 3. Start tunneling
zrok share reserved hybridcrs

# Other alternatives tested #

# https://dashboard.ngrok.com/get-started/setup/linux
ngrok http --url=https://perfectly-large-goshawk.ngrok-free.app 8000

# https://gitlab.com/pyjam.as/tunnel
curl https://tunnel.pyjam.as/{PORT} > tunnel.conf && wg-quick up ./tunnel.conf
wg-quick up ./tunnel.conf
wg-quick down ./tunnel.conf
