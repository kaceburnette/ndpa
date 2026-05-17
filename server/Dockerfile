FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY server/requirements.txt ./
RUN pip install -r requirements.txt

COPY server/ ./server/

EXPOSE 8000

# uvloop + httptools enabled via uvicorn[standard]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
