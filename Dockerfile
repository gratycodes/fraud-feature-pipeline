FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[kafka]"

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["ffp"]
CMD ["--help"]
