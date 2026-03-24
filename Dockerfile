FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
RUN pip install uv -i https://mirrors.aliyun.com/pypi/simple
RUN uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple 


COPY . .

RUN mkdir -p logs data/process data/chunk data/vector_database

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
