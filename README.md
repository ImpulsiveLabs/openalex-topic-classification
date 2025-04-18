# OpenAlex Topic Classification (Local Fork)

This repository is a fork of the [openalex-topic-classification](https://github.com/OurResearch/openalex-topic-classification) project by **OurResearch**, specifically based on the `predictor.py` file located at `v1/003_Deployment/model_to_api/container/topic_classifier/predictor.py`. It has been adapted to run locally without requiring external model downloads from Hugging Face, ensuring all necessary model files are included for offline operation.

## Overview

This project provides a Dockerized Flask-based inference server for topic classification of academic papers using the OpenAlex dataset. It leverages two pre-trained models:

- **sentence-transformers/all-MiniLM-L6-v2** for journal embeddings.
- **OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract** for title and abstract processing.

The fork prioritizes **local-first execution**, with all model files pre-downloaded and stored in the `model/` directory, eliminating runtime dependencies on external repositories.

## Features

- **Offline Model Execution**: Includes all Hugging Face models and supporting files (e.g., `.pkl`, `citation_part_only.keras`) for local inference.
- **Dockerized Setup**: Uses `Dockerfile` and `docker-compose.yml` for consistent deployment.
- **Flask Inference Server**: Serves predictions via a REST API, powered by Gunicorn and Nginx.
- **Complete Model Files**: Contains all required files for topic classification without internet access.
