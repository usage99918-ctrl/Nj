FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=60s --timeout=20s --start-period=180s --retries=3 CMD python -m cryptolab health || exit 1
CMD ["python", "-m", "cryptolab", "run"]
