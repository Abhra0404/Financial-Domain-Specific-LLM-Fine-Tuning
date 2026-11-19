# Financial Domain Adaptation and Fine-Tuning of Large Language Models

## Overview

This project investigates the effectiveness of domain adaptation and parameter-efficient fine-tuning techniques in improving the performance of general-purpose Large Language Models (LLMs) on financial natural language processing and reasoning tasks.

The primary objective is to study whether continued pretraining on financial-domain corpora, followed by supervised fine-tuning, can improve a model's ability to understand financial terminology, interpret financial documents, extract structured information, perform financial sentiment analysis, and answer questions involving numerical financial data.

Rather than developing another general-purpose financial chatbot, this project focuses on the underlying model adaptation process, experimental methodology, and quantitative evaluation of fine-tuned models.

## Research Motivation

General-purpose LLMs possess broad linguistic and reasoning capabilities but may struggle with domain-specific terminology, financial document structures, numerical relationships, and specialized financial reasoning tasks.

Financial language presents several challenges:

- Domain-specific terminology and abbreviations.
- Complex financial statements and reporting structures.
- Numerical reasoning involving multiple financial variables.
- Context-dependent financial sentiment.
- Long-form financial documents and earnings reports.

This project investigates whether domain-specific adaptation can address these limitations without requiring the computational resources associated with training an LLM from scratch.

## Research Question

**Can domain-adaptive pretraining and parameter-efficient fine-tuning improve the performance of general-purpose LLMs on financial language understanding, information extraction, and numerical reasoning tasks?**

### Research Objectives

1. Establish baseline performance of a pretrained general-purpose LLM on financial benchmarks.
2. Construct a curated financial-domain corpus for continued pretraining.
3. Investigate Domain-Adaptive Pretraining (DAPT).
4. Implement Supervised Fine-Tuning (SFT) using financial instruction datasets.
5. Compare LoRA and QLoRA fine-tuning strategies.
6. Investigate the combined effect of DAPT followed by SFT.
7. Evaluate adapted models using task-specific quantitative metrics.
8. Analyze computational efficiency, inference latency, and memory consumption.

## Research Scope

The project focuses on four major financial NLP capabilities.

| Capability | Description |
|---|---|
| Financial Sentiment Analysis | Classifying financial statements and news into positive, negative, and neutral sentiment |
| Financial Question Answering | Answering questions based on financial reports and documents |
| Financial Information Extraction | Extracting financial entities, metrics, and structured numerical information |
| Financial Summarization | Producing concise summaries of financial documents and earnings reports |

Numerical reasoning will be evaluated separately from general financial question answering to distinguish language understanding from computational correctness.

## Proposed Methodology

The research follows a controlled experimental approach.

### 1. Baseline Evaluation

Evaluate an unmodified pretrained language model on selected financial NLP benchmarks.

This establishes the reference performance against which all adapted models will be compared.

### 2. Domain-Adaptive Pretraining (DAPT)

Continue training the pretrained model using an unlabeled financial-domain corpus.

Potential sources include:

- SEC EDGAR filings.
- Annual reports.
- Financial statements.
- Earnings call transcripts.
- Financial research documents.

The objective is to investigate whether exposure to financial-domain language improves domain-specific understanding.

### 3. Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model using financial instruction-response datasets.

Training tasks include:

- Financial question answering.
- Financial sentiment classification.
- Structured information extraction.
- Financial summarization.

### 4. Parameter-Efficient Fine-Tuning

Investigate and compare:

- LoRA (Low-Rank Adaptation).
- QLoRA (Quantized Low-Rank Adaptation).

The experiments will investigate trainable parameter counts, memory requirements, training time, and downstream performance.

### 5. Combined Domain Adaptation

Investigate a sequential adaptation pipeline:

Pretrained Model → DAPT → SFT → Financial Evaluation

Compare this against direct SFT without DAPT.

## Experimental Design

The initial experimental matrix consists of:

| Experiment | Training Strategy |
|---|---|
| E0 | Original pretrained model |
| E1 | SFT with LoRA |
| E2 | SFT with QLoRA |
| E3 | DAPT followed by SFT |
| E4 | Comparative evaluation and ablation analysis |

Where computationally feasible, additional experiments will compare conventional LoRA and QLoRA under controlled configurations.

## Model Selection

The primary research model will initially be:

**Qwen2.5-3B Base**

Model repository: https://huggingface.co/Qwen/Qwen2.5-3B

Reasons for selection:

- Relatively manageable parameter count.
- Suitable for controlled fine-tuning experiments.
- Open-weight availability.
- Compatibility with common Hugging Face training workflows.
- Practical deployment requirements compared with larger models.

A smaller model, Qwen2.5-1.5B-Instruct, will be used for local experimentation on Apple Silicon.

Model selection and licensing will be verified before training.

## Dataset Sources

Potential datasets include:

| Dataset | Application |
|---|---|
| Financial PhraseBank | Financial sentiment analysis |
| FiQA | Financial question answering and sentiment |
| FinQA | Numerical financial reasoning |
| ConvFinQA | Conversational financial reasoning |
| SEC EDGAR | Financial-domain pretraining corpus |

Dataset licenses, provenance, splits, and usage restrictions will be examined before inclusion.

## Evaluation Methodology

Different tasks will be evaluated using appropriate metrics.

| Task | Evaluation Metrics |
|---|---|
| Sentiment Analysis | Accuracy, Precision, Recall, F1-score |
| Financial QA | Exact Match, Token-level F1 |
| Numerical Reasoning | Numerical Accuracy, Execution Accuracy |
| Information Extraction | Precision, Recall, F1-score |
| Summarization | ROUGE-L, Human Evaluation |

Additional system-level measurements:

- Training duration.
- Peak GPU memory consumption.
- Trainable parameter count.
- Model artifact size.
- Inference latency.
- Throughput.

All models will be evaluated against identical held-out test sets wherever the task and dataset permit.

## Technology Stack

### Machine Learning
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

### Model Serving
- FastAPI
- Docker

### Development Environment
- Apple Silicon MacBook
- Google Colab / NVIDIA GPU cloud instances
- Jupyter Notebook
- Git and GitHub

## Hardware Strategy

Local development and lightweight experiments will be performed on an Apple Silicon M4 MacBook with 16 GB unified memory.

Cloud GPU resources will be used for computationally intensive experiments.

Proposed cloud hardware:

- NVIDIA L4 (24 GB VRAM) for smaller experiments.
- NVIDIA A100 (40 GB VRAM) for primary fine-tuning experiments.

The final hardware configuration will depend on model memory requirements, sequence length, and training strategy.

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
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_baseline_evaluation.ipynb
│   └── 04_experiment_analysis.ipynb
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
│   ├── dapt/
│   ├── lora/
│   └── qlora/
│
├── models/
│
├── api/
│
├── tests/
│
├── docs/
│   ├── research/
│   ├── experiments/
│   └── results/
│
├── requirements.txt
├── README.md
├── PLAN.md
└── .gitignore
```

Model checkpoints and large datasets will be maintained outside Git and referenced through documented artifact locations.

## Expected Contributions

The expected outcome is a reproducible experimental study and engineering implementation that provides:

1. A curated financial-domain dataset pipeline.
2. Baseline evaluation of a general-purpose LLM.
3. Implementations of DAPT, SFT, LoRA, and QLoRA.
4. Comparative benchmark results.
5. An analysis of domain adaptation effectiveness.
6. Computational efficiency comparisons.
7. Reproducible training and evaluation configurations.
8. An inference interface for demonstrating the adapted model.

These are intended contributions; their effectiveness will be established through experiments rather than assumed in advance.

## Research Principles

- Reproducibility of experiments.
- Dataset provenance and license compliance.
- Separation of training, validation, and test data.
- Controlled experimental comparisons.
- Transparent reporting of negative as well as positive results.
- No unsupported claims about model performance.

## Limitations

Financial language understanding does not automatically imply financial expertise or reliable investment advice.

Numerical outputs must be independently validated. The model may generate incorrect information, and the project is intended for research and educational purposes rather than production financial decision-making.

## Project Status

**Status:** Research planning and environment preparation.

Implementation will proceed incrementally, beginning with literature review, dataset investigation, and baseline model evaluation.