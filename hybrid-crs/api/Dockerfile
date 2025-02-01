# BUILD IMAGE:       docker build --network=host -t hybrid-crs-api .
# RUN CONTAINER:     docker run --name hybrid-crs-api -p 8000:8000 hybrid-crs-api
# STOP CONTAINER:    docker stop hybrid-crs-api
# RESUME CONTAINER:  docker start hybrid-crs-api

FROM python:3.12.8-alpine3.21 AS base

# Install base dependencies
RUN apk add --update --no-cache bash postgresql

# Environment variables
ENV PGSERVICEFILE=/app/storm/.pg_service.conf

WORKDIR /app

# --- First Stage: Install project dependencies ---
FROM base AS deps

# Backend code
ADD storm /app/storm
ADD requirements.txt /app/

# Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage: Run DB and Backend servers ---
FROM deps AS final

# Initialize database
USER postgres
RUN mkdir /var/lib/postgresql/data \
    && chmod 0700 /var/lib/postgresql/data \
    && initdb /var/lib/postgresql/data \
    && echo "unix_socket_directories = '/tmp'" >> /var/lib/postgresql/data/postgresql.conf
USER root

# Volume for DB data
VOLUME ["/var/lib/postgresql/data"]

# Expose port
EXPOSE 8000

# Initialize servers
CMD ["fastapi", "run", "main.py"]
