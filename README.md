# Hybrid-CRS
Master's Thesis project, a platform to create Hybrid Conversational Recommender System Agents by leveraging LLMs. Uses Knowledge Graphs and RecBole models to obtain recommendations, as well as tool-calling capabilities to communicate with external functions and carry out the conversations. The platform includes functionalities for chatting with LLMs freely, with voice chat and web search capabilities.

Deployed at https://hybrid-crs.vercel.app, with local backend proxy using [zrok](https://zrok.io/).

- Frontend Repository: [Hybrid-CRS-Frontend](https://github.com/Acervans/Hybrid-CRS-Frontend)
- Thesis Repository: [Hybrid-CRS-Thesis](https://github.com/Acervans/Hybrid-CRS-Thesis)

Backend Tech Stack:
- `FastAPI` as the RESTful API entrypoint
- `FalkorDB` as the graph store for recommendation data and chat history
- `RecBole` as the recommendation framework for expert models
- `Ollama` as the LLM server
- `LlamaIndex` as the agent orchestrator
- `Supabase` as the cloud-based metadata database

## Setup
1. Install [`Docker`](https://www.docker.com/get-started/) and [`Docker Compose`](https://docs.docker.com/compose/).
2. Configure the [`NVIDIA Container Runtime`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) to work with `Docker`:
   - If you want to run the `Ollama` service with AMD GPUs or CPU only, check the [Docker image](https://hub.docker.com/r/ollama/ollama#ollama-docker-image).
```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
3. Create a [`Supabase`](https://supabase.com/) project using the __roles.sql__ and __schema.sql__ files in `supabase/`, following [this guide](https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore#restore-backup-using-cli) (without __data.sql__).
4. In `hybrid-crs/`:
   - Create a __.env__ file using __.env.example__ as sample. Replace with Supabase JWT secret, service role key and URL.
   - In `frontend/hybrid-crs-ui/`, create a __.env.local__ file using __.env.local.example__ as sample. Replace with Supabase anonymous key and URL.
   - Run `docker compose up` to pull and build all the images for the project, and start the Ollama, FalkorDB, API and UI services.
5. All set! The frontend can be accessed at http://localhost:3001.
   - Once signed up and authenticated, use the LLM Selector (top right), and pull the `qwen2.5:3b` model, required for the project.

### Developer Setup
1. Go to `hybrid-crs/` and install Python dependencies in __requirements-dev.txt__. Using __pip__:
```shell
pip install -r requirements-dev.txt
```
2. In root directory, install pre-commit hooks in the repository with:
```shell
pre-commit install
```
3. To manually start the backend server (API service in Docker Compose):
   - Go to `hybrid-crs/`.
   - Run the FastAPI server:
```shell
# Development mode
fastapi dev api.py
```
```shell
# Production mode
fastapi run api.py
```
4. To manually start the frontend server (UI service in Docker Compose):
   - Go to `hybrid-crs/frontend/hybrid-crs-ui/`.
   - Install packages with your package manager.
   - Run the Next.js app. Using __pnpm__:
```shell
# Install packages
pnpm install
```
```shell
# Development mode
pnpm dev
```
```shell
# Production mode
pnpm build:dev
pnpm start
```
