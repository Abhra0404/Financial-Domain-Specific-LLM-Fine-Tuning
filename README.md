# Financial Domain-Specific LLM Fine-Tuning

An end-to-end Large Language Model engineering project focused on adapting open-source LLMs to financial-domain tasks through supervised fine-tuning and parameter-efficient fine-tuning techniques.

## Overview

This project aims to develop a financial-domain-adapted Large Language Model capable of understanding financial terminology, processing financial documents, answering domain-specific questions, extracting structured financial information, and performing financial sentiment analysis.

The implementation focuses on the complete LLM engineering lifecycle, including dataset engineering, model selection, supervised fine-tuning, parameter-efficient adaptation, model evaluation, inference optimization, and deployment.

Rather than building a generic financial chatbot using an external LLM API, the project involves fine-tuning an open-source pretrained model and deploying the resulting model through a dedicated inference service.

## Objectives

- Adapt a general-purpose pretrained LLM to financial-domain tasks.
- Build a reproducible financial dataset preparation pipeline.
- Implement Supervised Fine-Tuning (SFT).
- Apply LoRA and QLoRA for parameter-efficient fine-tuning.
- Evaluate fine-tuned models against pretrained baselines.
- Optimize model inference and memory consumption.
- Develop a FastAPI-based inference service.
- Build an interactive financial document analysis application.
- Containerize and deploy the complete system.

## Core Capabilities

### 1. Financial Question Answering

Answer questions based on financial documents, reports, statements, and supplied financial context.

Example:

**Input:**
What was the company's revenue growth between the two reported financial years?

**Expected Output:**
A structured answer containing the relevant financial figures, calculation, and explanation.

### 2. Financial Sentiment Analysis

Classify financial statements, news excerpts, and market-related text into:

- Positive
- Negative
- Neutral

### 3. Financial Information Extraction

Extract structured financial information from unstructured documents.

Supported information may include:

- Revenue
- Net income
- Operating expenses
- Earnings per share (EPS)
- Total assets
- Total liabilities
- Debt
- Profit margins

Outputs will follow a predefined JSON schema.

### 4. Financial Document Summarization

Generate structured summaries of financial documents, including:

- Financial performance.
- Revenue and profitability.
- Important financial metrics.
- Significant changes.
- Potential risk factors.
- Key management observations.

### 5. Numerical Financial Reasoning

Answer questions involving numerical information from financial reports.

The system will distinguish model-generated reasoning from independently validated numerical calculations.

## Technical Architecture

```text
                  Financial Datasets
                         |
                         v
                Data Engineering
                         |
             Cleaning and Transformation
                         |
                         v
                  Tokenization
                         |
                         v
                Pretrained LLM
                         |
                         v
              Fine-Tuning Pipeline
                         |
                 +-------+-------+
                 |               |
                LoRA           QLoRA
                 |               |
                 +-------+-------+
                         |
                         v
                 Fine-Tuned Model
                         |
                         v
               Model Evaluation
                         |
                         v
                 Model Optimization
                         |
                         v
                 FastAPI Backend
                         |
                         v
                Financial AI UI
                         |
                         v
                  Docker Deployment
```

## Model Selection

### Primary Model

**Qwen2.5-3B Base**

Model: https://huggingface.co/Qwen/Qwen2.5-3B

The model will serve as the primary candidate for cloud-based fine-tuning experiments.

Selection considerations:

- Manageable parameter count.
- Open-weight availability.
- Compatibility with Hugging Face Transformers.
- Suitability for parameter-efficient fine-tuning.
- Practical inference requirements.

### Local Development Model

**Qwen2.5-1.5B-Instruct**

Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Used for local inference, pipeline validation, and lightweight fine-tuning experiments on Apple Silicon.

Model licenses and architecture compatibility will be verified before implementation.

## Fine-Tuning Methodology

### Supervised Fine-Tuning (SFT)

The primary adaptation method will be Supervised Fine-Tuning using curated financial instruction-response datasets.

Training tasks will include:

- Financial question answering.
- Sentiment classification.
- Information extraction.
- Financial summarization.

### LoRA

Low-Rank Adaptation will be used to fine-tune selected model parameters through trainable low-rank adapter matrices.

Benefits:

- Reduced trainable parameter count.
- Lower memory requirements than full fine-tuning.
- Smaller model adaptation artifacts.
- Convenient experimentation.

### QLoRA

Quantized Low-Rank Adaptation will be investigated using a quantized base model and trainable low-rank adapters.

The objective is to reduce training memory requirements while maintaining useful downstream performance.

### Domain-Adaptive Pretraining (Advanced)

As an optional enhancement, continued pretraining on financial-domain text may be implemented before supervised fine-tuning.

This will investigate whether exposure to financial documents improves downstream domain-specific capabilities.

## Dataset Sources

| Dataset | Purpose |
|---|---|
| Financial PhraseBank | Financial sentiment classification |
| FiQA | Financial sentiment and question answering |
| FinQA | Numerical financial reasoning |
| ConvFinQA | Conversational financial question answering |
| SEC EDGAR | Financial document collection |

Dataset licenses, provenance, usage restrictions, and benchmark splits will be reviewed before use.

## Technology Stack

### Machine Learning and Fine-Tuning

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT
- TRL
- BitsAndBytes
- MLX / MLX-LM

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Experiment Tracking

- Weights & Biases

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- React
- JavaScript
- Tailwind CSS

### Infrastructure

- Docker
- Git
- GitHub
- NVIDIA Cloud GPUs

## Hardware Strategy

### Local Development

Apple MacBook M4 with 16 GB unified memory and 512 GB storage.

Local responsibilities:

- Dataset exploration and preprocessing.
- Tokenization.
- Baseline inference.
- Lightweight MLX-LM experiments.
- Evaluation pipeline development.
- Backend and frontend development.

### Cloud Training

Cloud GPUs will be used for computationally intensive training.

Preferred hardware:

- NVIDIA L4 (24 GB VRAM).
- NVIDIA A100 (40 GB VRAM).

The GPU configuration will be selected according to model size, sequence length, batch size, and memory requirements.

## Evaluation

Fine-tuned models will be compared against their original pretrained baselines.

| Task | Metrics |
|---|---|
| Sentiment Analysis | Accuracy, Precision, Recall, F1 |
| Financial QA | Exact Match, Token F1 |
| Information Extraction | Precision, Recall, F1 |
| Numerical Reasoning | Numerical Accuracy, Execution Accuracy |
| Summarization | ROUGE-L, Human Evaluation |

Additional system measurements:

- Training duration.
- GPU memory consumption.
- Trainable parameter count.
- Adapter size.
- Inference latency.
- Throughput.

## API Capabilities

The FastAPI inference service will expose the following endpoints:

| Endpoint | Function |
|---|---|
| `/health` | Service health |
| `/predict/sentiment` | Financial sentiment analysis |
| `/predict/qa` | Financial question answering |
| `/predict/extraction` | Financial information extraction |
| `/predict/summarization` | Financial document summarization |

## Repository Structure

```text
financial-domain-llm/
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── experiment_config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_baseline_evaluation.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── experiments/
│   ├── baseline/
│   ├── lora/
│   ├── qlora/
│   └── dapt/
│
├── models/
│
├── api/
│
├── frontend/
│
├── tests/
│
├── docs/
│   ├── implementation/
│   ├── experiments/
│   └── results/
│
├── requirements.txt
├── README.md
├── PLAN.md
└── .gitignore
```

Large datasets, pretrained checkpoints, and generated model artifacts will be excluded from Git and managed separately.

## Expected Deliverables

1. Financial dataset engineering pipeline.
2. Pretrained baseline model and evaluation pipeline.
3. Financial-domain SFT implementation.
4. LoRA and QLoRA adapters.
5. Comparative model evaluation.
6. Optimized inference pipeline.
7. FastAPI model-serving backend.
8. Interactive financial AI application.
9. Dockerized deployment.
10. Technical documentation and reproducible configurations.

## Project Limitations

- Model outputs may contain incorrect financial information.
- Numerical outputs require independent verification.
- Fine-tuning does not guarantee reliable investment advice.
- Financial data quality directly influences model performance.
- Performance improvements must be established through evaluation rather than assumed.

This project is intended for educational and engineering demonstration purposes, not production financial decision-making.

## Current Status

**Status:** Planning and environment preparation.

**Next Milestone:** Local environment setup, financial dataset investigation, and baseline model implementation.