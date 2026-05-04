# Match requirements.txt: pinned versions (numpy 1.26.4, torch 2.2.2) target Python 3.11.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build toolchain for any wheels that need to compile (e.g. hmmlearn) plus git for VCS deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the dashboard by default
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
