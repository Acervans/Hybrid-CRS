# BUILD IMAGE:       docker build --network=host -t hybrid-crs-api .
# RUN CONTAINER:     docker run --name hybrid-crs-api -p 8000:8000 hybrid-crs-api
# STOP CONTAINER:    docker stop hybrid-crs-api
# RESUME CONTAINER:  docker start hybrid-crs-api

FROM python:3.12.11-slim-bookworm AS base

WORKDIR /api

# --- Dependencies Stage ---
FROM base AS deps

# Add Backend code
COPY api.py schemas.py /api/

COPY data_processing/data_utils.py /api/data_processing/
COPY \
    llm/falkordb_chat_history.py \
    llm/hybrid_crs_workflow.py \
    llm/user_profile.py \
    /api/llm/

COPY recsys/config /api/recsys/config
COPY \
    recsys/falkordb_recommender.py \
    recsys/recbole_utils.py \
    /api/recsys/

# Python dependencies
ADD requirements.txt /api/
RUN pip install --no-cache-dir -r requirements.txt

# --- Deployment Stage ---
FROM deps AS final

# Expose port
EXPOSE 8000

# Initialize servers
CMD ["fastapi", "run", "api.py"]
