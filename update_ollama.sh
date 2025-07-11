cd hybrid-crs

# First stop the service
docker compose down ollama

# Reload nvidia_uvm
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm

# Then restart the service
docker-compose up --force-recreate --build -d ollama
docker image prune -f

cd -
