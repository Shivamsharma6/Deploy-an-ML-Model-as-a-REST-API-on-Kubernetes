FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app + model
COPY app.py .
COPY iris_model.pkl .

CMD ["python", "app.py"]
