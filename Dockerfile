FROM python:3.9

ENV UPLOAD_DIR=/uploads
ENV DB_PATH=/data/db.sqlite3
ENV PATH=/home/user/.local/bin:$PATH

RUN apt-get update && \
    apt-get install -y \
        libmagic1 \
        ffmpeg \
        gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m --uid=1000 -o user

ADD ./requirements.txt /tmp/requirements.txt
RUN gosu user pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir /uploads /data && \
    chown -R 1000:1000 /uploads /data

WORKDIR /code
ADD ./main.py /code/main.py

CMD ["gosu", "user", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
