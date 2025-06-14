FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --default-timeout=300 -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
