FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + submission
COPY main.py .
COPY submission.csv .

# HuggingFace Spaces runs on port 7860
ENV SUBMISSION_PATH=/app/submission.csv
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
