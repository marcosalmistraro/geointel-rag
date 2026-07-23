FROM python:3.11-slim

WORKDIR /app

# libgeos-dev is needed by shapely (geospatial geometry library).
# Newer shapely/pyogrio wheels bundle their own GDAL so no libgdal-dev needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying code so this layer is cached
# and only re-runs when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
