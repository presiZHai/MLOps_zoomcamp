FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# Copy your Python script into the Docker image
COPY scoring.py .

# Run the Python script
CMD ["python", "scoring.py", "2023", "5"]