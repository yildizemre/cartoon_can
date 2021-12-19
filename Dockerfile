FROM python:3-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN set -eux; \
        apt-get update; \
        apt-get install -y --no-install-recommends gcc; \
        apt-get install -y  python3-dev; \
        apt-get install -y libevent-dev; \
        pip install --no-cache-dir py-zipkin; \
        apt-get -y  autoremove gcc; \
        rm -rf /var/lib/apt/lists/*


RUN set -eux; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
                libgl1-mesa-glx \
                ffmpeg \
        ; \
        rm -rf /var/lib/apt/lists/*

COPY ./ /usr/src/app

CMD ["uvicorn", hypegenai,"0.0.0.0"]
