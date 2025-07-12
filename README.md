# Hybrid-CRS
Master's Thesis project, a platform to create Hybrid Conversational Recommender Systems leveraging LLMs. Provides Retrieval Augmented Generation to obtain recommendable items, and tool-calling capabilities to communicate with external functions such as specific recommendation models.

- Frontend Repository: [Hybrid-CRS-Frontend](https://github.com/Acervans/Hybrid-CRS-Frontend)
- Thesis Repository: [Hybrid-CRS-Thesis](https://github.com/Acervans/Hybrid-CRS-Thesis)


## Setup
1. Install [`Docker`](https://www.docker.com/get-started/) and [`Docker Compose`](https://docs.docker.com/compose/).
2. Create a [`Supabase`](https://supabase.com/) project using the __roles.sql__ and __schema.sql__ files in `supabase/`, following [this guide](https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore#restore-backup-using-cli) (without __data.sql__).
3. In `frontend/hybrid-crs-ui/`, create a __.env.local__ file using __.env.local.example__ as sample. Replace with Supabase anonymous key and URL.
4. In `hybrid-crs/`:
   - Create a __.env__ file using __.env.example__ as sample. Replace with Supabase JWT secret, service role key and URL.
   - Run `docker compose up` to pull and build all the images for the project, and start the Ollama, FalkorDB, API and UI services.
5. All set! The frontend can be accessed at http://localhost:3001.

### Developer Setup
1. In `hybrid-crs/`, install Python dependencies in __requirements-dev.txt__. Using __pip__:
```shell
pip install -r requirements-dev.txt
```
2. Install pre-commit hooks in the repository with:
```shell
pre-commit install
```
3. To manually start the backend server (API service in Docker Compose), go to `hybrid-crs/` and start the FastAPI server:
```shell
fastapi dev api.py
```
4. To manually start the frontend server (UI service in Docker Compose), go to `frontend/hybrid-crs-ui/`, install packages and start the Next.js app with your package manager. Using __pnpm__:
```shell
pnpm install
pnpm dev
```
