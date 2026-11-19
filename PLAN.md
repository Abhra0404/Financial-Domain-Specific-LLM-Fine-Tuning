# Implementation Plan

## Financial Domain-Specific LLM Fine-Tuning

**Project Type:** Applied AI / Machine Learning Engineering

**Primary Focus:** LLM Fine-Tuning, PEFT, Model Evaluation, Optimization, and Deployment.

---

# 1. Project Goal

Build an end-to-end financial-domain-adapted LLM by fine-tuning an existing open-source language model.

The model should be capable of performing specialized financial NLP tasks, including question answering, sentiment analysis, financial information extraction, summarization, and numerical reasoning.

The implementation will emphasize practical ML engineering, reproducible training pipelines, measurable model improvements, and deployment.

# 2. Development Strategy

The project will follow this implementation sequence:

```text
Environment Setup
       |
       v
Dataset Engineering
       |
       v
Baseline Model
       |
       v
Supervised Fine-Tuning
       |
       v
LoRA and QLoRA
       |
       v
Model Evaluation
       |
       v
Inference Optimization
       |
       v
FastAPI Backend
       |
       v
React Application
       |
       v
Deployment
```

Domain-Adaptive Pretraining will be treated as an advanced extension after the primary SFT pipeline works.

# 3. Phase 1 — Environment and Infrastructure Setup

## Objective

Prepare the local development environment and cloud training environment.

### Local Environment

Hardware:

- Apple M4.
- 16 GB unified memory.
- 512 GB SSD.

Software:

- Python 3.11.
- MLX.
- MLX-LM.
- Hugging Face Transformers.
- Hugging Face Datasets.
- Pandas.
- NumPy.
- Scikit-learn.
- JupyterLab.
- Weights & Biases.

### Cloud Environment

Use an NVIDIA GPU instance for intensive model training.

Preferred GPU:

- A100 40 GB.
- L4 24 GB as a more economical alternative.

Software:

- PyTorch.
- CUDA-compatible Transformers.
- PEFT.
- TRL.
- BitsAndBytes.
- Accelerate.
- Weights & Biases.

### Tasks

1. Create the GitHub repository.
2. Set up the project directory.
3. Configure Python virtual environments.
4. Install local dependencies.
5. Verify MLX acceleration.
6. Prepare cloud GPU dependencies.
7. Verify GPU and CUDA availability.
8. Configure experiment tracking.
9. Establish artifact storage conventions.

### Deliverables

- Working local environment.
- Working cloud environment.
- Initial repository structure.
- Dependency configuration.
- Hardware verification documentation.

---

# 4. Phase 2 — Financial Dataset Engineering

## Objective

Collect, understand, clean, and prepare financial datasets for model training.

### Dataset Sources

**Financial PhraseBank**

Used for financial sentiment classification.

**FiQA**

Used for financial sentiment and question answering.

**FinQA**

Used for numerical financial reasoning.

**ConvFinQA**

Used for conversational financial question answering.

**SEC EDGAR**

Used as a potential source of financial documents for summarization and advanced domain adaptation.

### Tasks

1. Download permitted datasets.
2. Inspect dataset schemas.
3. Analyze sample records.
4. Identify missing values.
5. Remove duplicates.
6. Normalize text.
7. Identify malformed records.
8. Standardize data formats.
9. Convert examples into task-specific schemas.
10. Create reproducible train, validation, and test splits.

### Dataset Formatting

Instruction-based examples should follow a consistent format.

Example:

```json
{
  "instruction": "Analyze the financial sentiment.",
  "input": "The company reported a significant increase in annual revenue.",
  "output": "Positive"
}
```

Question-answering examples should preserve their supporting financial context.

Extraction examples should use consistent structured output schemas.

### Data Quality Checks

- Duplicate detection.
- Invalid example filtering.
- Label distribution analysis.
- Token-length analysis.
- Source document tracking.
- Train/test contamination checks.

### Deliverables

- Dataset ingestion scripts.
- Cleaning and transformation pipeline.
- Standardized training datasets.
- Dataset documentation.
- Data quality report.

---

# 5. Phase 3 — Baseline Model Implementation

## Objective

Load a pretrained model and establish its performance before fine-tuning.

### Model Selection

Primary cloud model:

Qwen2.5-3B Base.

Local development model:

Qwen2.5-1.5B-Instruct.

### Tasks

1. Load the pretrained model.
2. Load the corresponding tokenizer.
3. Verify model inference.
4. Prepare evaluation prompts.
5. Implement task-specific output parsing.
6. Run baseline evaluations.
7. Save predictions.
8. Calculate performance metrics.

### Evaluation

- Financial sentiment classification.
- Financial question answering.
- Financial information extraction.
- Numerical financial reasoning.
- Financial summarization.

### Deliverables

- Baseline inference scripts.
- Evaluation pipeline.
- Baseline metrics.
- Prediction artifacts.
- Model configuration.

The baseline must be established before fine-tuning so that later improvements can be measured accurately.

---

# 6. Phase 4 — Supervised Fine-Tuning

## Objective

Adapt the pretrained model to perform financial-domain tasks using labeled instruction-response data.

### Training Framework

- PyTorch.
- Hugging Face Transformers.
- TRL.
- Hugging Face Datasets.

### Training Tasks

1. Financial question answering.
2. Financial sentiment analysis.
3. Financial information extraction.
4. Financial summarization.

### Tasks

1. Prepare instruction datasets.
2. Apply the model's chat template where applicable.
3. Configure tokenization.
4. Prepare training and validation datasets.
5. Configure supervised fine-tuning.
6. Train an initial adapter.
7. Monitor training and validation loss.
8. Save checkpoints.
9. Run evaluation.

### Initial Training Configuration

| Parameter | Value |
|---|---|
| Model | Qwen2.5-3B Base |
| Training Method | SFT |
| Sequence Length | 2048 |
| Precision | BF16 where supported |
| Learning Rate | 2e-4 starting point |
| Epochs | 2–3 |
| Scheduler | Cosine |
| Warmup Ratio | 0.03 |

These settings will be validated through pilot runs and adjusted according to the dataset and hardware.

### Deliverables

- SFT training pipeline.
- Fine-tuned model adapter.
- Training logs.
- Evaluation results.

---

# 7. Phase 5 — Parameter-Efficient Fine-Tuning

## Objective

Implement and compare LoRA and QLoRA.

### A. LoRA

Implement Low-Rank Adaptation.

Study and configure:

- Rank.
- Alpha.
- Dropout.
- Target modules.
- Trainable parameters.
- Adapter saving and loading.

Initial configuration:

| Parameter | Value |
|---|---|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Learning Rate | 2e-4 |

### B. QLoRA

Implement quantized fine-tuning.

Study:

- 4-bit quantization.
- NF4.
- Double quantization.
- Compute dtype.
- Quantized model loading.
- Memory optimization.

### Tasks

1. Train LoRA adapters.
2. Train QLoRA adapters.
3. Measure training memory.
4. Measure training duration.
5. Compare adapter sizes.
6. Compare downstream performance.
7. Record experiment configurations.

### Deliverables

- LoRA implementation.
- QLoRA implementation.
- Adapter artifacts.
- Experiment comparison report.

---

# 8. Phase 6 — Advanced Domain Adaptation

## Objective

Investigate whether continued pretraining on financial text improves the model's downstream performance.

This phase is an advanced enhancement and should only begin after the SFT pipeline is stable.

### Domain-Adaptive Pretraining

Use a curated corpus containing financial-domain text.

Potential sources:

- SEC filings.
- Financial statements.
- Annual reports.
- Earnings transcripts.

### Tasks

1. Prepare the financial corpus.
2. Remove duplicate and low-quality content.
3. Tokenize documents.
4. Configure causal language modeling.
5. Continue pretraining the selected base model.
6. Monitor validation loss.
7. Save the adapted checkpoint.
8. Apply supervised fine-tuning.
9. Evaluate the resulting model.

### Comparison

Compare:

- Original model + SFT.
- DAPT model + SFT.

Also monitor general-domain performance to detect catastrophic forgetting.

### Deliverables

- DAPT pipeline.
- Domain-adapted checkpoint.
- DAPT training logs.
- Downstream evaluation comparison.

---

# 9. Phase 7 — Model Evaluation and Benchmarking

## Objective

Measure the effectiveness of the fine-tuned model.

### Metrics

| Capability | Metrics |
|---|---|
| Sentiment Analysis | Accuracy, Precision, Recall, F1 |
| Financial QA | Exact Match, Token F1 |
| Information Extraction | Precision, Recall, F1 |
| Numerical Reasoning | Numerical Accuracy, Execution Accuracy |
| Summarization | ROUGE-L, Human Evaluation |

### System-Level Metrics

- Training time.
- GPU memory consumption.
- Trainable parameter count.
- Adapter size.
- Inference latency.
- Throughput.

### Error Analysis

Identify:

- Incorrect financial figures.
- Hallucinated information.
- Incorrect calculations.
- Missing extracted fields.
- Incorrect sentiment.
- Unsupported summaries.

### Deliverables

- Benchmarking scripts.
- Comparative evaluation tables.
- Performance visualizations.
- Error analysis report.

---

# 10. Phase 8 — Model Optimization and Inference

## Objective

Prepare the adapted model for efficient inference.

### Tasks

1. Load trained adapters.
2. Test adapter merging where supported.
3. Investigate model quantization.
4. Measure inference latency.
5. Measure memory consumption.
6. Compare inference configurations.
7. Implement model loading and caching.
8. Optimize response generation.

### Deliverables

- Optimized model configuration.
- Inference benchmarks.
- Model artifact documentation.

---

# 11. Phase 9 — Backend Development

## Objective

Expose the fine-tuned model through a production-oriented inference API.

### Technology

- FastAPI.
- Python.
- Pydantic.
- Uvicorn.

### API Endpoints

| Endpoint | Responsibility |
|---|---|
| GET `/health` | Health check |
| POST `/predict/sentiment` | Financial sentiment |
| POST `/predict/qa` | Financial question answering |
| POST `/predict/extraction` | Structured extraction |
| POST `/predict/summarization` | Document summarization |

### Features

- Model loading and lifecycle management.
- Input validation.
- Structured responses.
- Error handling.
- Request logging.
- Inference timing.
- Configurable model selection.

### Deliverables

- FastAPI application.
- Request and response schemas.
- API documentation.
- Inference integration tests.

---

# 12. Phase 10 — Frontend Development

## Objective

Build a clean interface demonstrating the model's financial capabilities.

### Technology

- React.
- JavaScript.
- Tailwind CSS.

### Main Interface Modules

**1. Dashboard**

Overview of the available financial AI capabilities.

**2. Financial Assistant**

Ask financial questions using supplied document context.

**3. Sentiment Analysis**

Submit financial statements or news text for classification.

**4. Document Analysis**

Upload financial documents and extract structured information.

**5. Summarization**

Generate summaries of financial reports.

**6. Model Information**

Display the active model, adaptation method, and available benchmark metrics.

### UX Requirements

- Clear input and output hierarchy.
- Loading and error states.
- Structured financial outputs.
- Copyable results.
- Responsive layouts.
- Transparent model limitations.

### Deliverables

- React application.
- API integration.
- Financial AI dashboard.

---

# 13. Phase 11 — Deployment

## Objective

Package the complete application for reproducible execution.

### Components

- Model inference service.
- FastAPI backend.
- React frontend.
- Docker configuration.

### Tasks

1. Create Dockerfiles.
2. Configure environment variables.
3. Configure model artifact locations.
4. Connect the frontend and backend.
5. Add service health checks.
6. Test complete application workflows.
7. Document deployment instructions.

### Deliverables

- Dockerized application.
- Deployment instructions.
- Environment configuration.
- Demonstration deployment.

---

# 14. Phase 12 — Final Documentation

## Objective

Document the engineering implementation and its measured results.

### Documentation

- System architecture.
- Dataset pipeline.
- Model selection.
- Fine-tuning implementation.
- LoRA and QLoRA comparison.
- Evaluation methodology.
- Performance benchmarks.
- Inference optimization.
- Deployment instructions.
- Known limitations.

### Final Deliverables

- Complete GitHub repository.
- Reproducible training scripts.
- Fine-tuned model adapters.
- Evaluation results.
- Inference API.
- Frontend application.
- Docker deployment.
- Technical documentation.

---

# 15. Suggested Timeline

| Phase | Duration |
|---|---|
| Environment Setup | 2–3 days |
| Dataset Engineering | 5–7 days |
| Baseline Implementation | 3–4 days |
| SFT Implementation | 5–7 days |
| LoRA and QLoRA | 4–5 days |
| Advanced DAPT | 5–7 days |
| Evaluation and Optimization | 4–5 days |
| Backend Development | 3–4 days |
| Frontend Development | 4–5 days |
| Deployment and Documentation | 3–4 days |

**Estimated duration: 6–8 weeks** for a focused implementation, with additional time for advanced DAPT experiments.

---

# 16. Completion Criteria

The project will be considered complete when:

- [ ] Financial dataset pipeline is implemented.
- [ ] Pretrained baseline is evaluated.
- [ ] Financial SFT is implemented.
- [ ] LoRA fine-tuning is implemented.
- [ ] QLoRA fine-tuning is implemented.
- [ ] Model performance is benchmarked.
- [ ] Inference optimization is measured.
- [ ] FastAPI inference service is operational.
- [ ] React frontend is integrated.
- [ ] Docker deployment works.
- [ ] Documentation and setup instructions are complete.

Advanced DAPT is an additional milestone rather than a blocker for the core application.

---

# 17. Development Principles

1. Prioritize the actual ML model over the user interface.
2. Understand the fine-tuning process instead of blindly executing scripts.
3. Keep training and inference code modular.
4. Maintain reproducible experiment configurations.
5. Never claim performance improvements without measurements.
6. Track compute costs and memory usage.
7. Keep model checkpoints outside Git.
8. Validate numerical outputs independently.
9. Document limitations and failure cases.
10. Build incrementally and maintain a working application at every major milestone.

**Final Goal:** Deliver a complete, fine-tuned financial-domain LLM with measurable performance, an optimized inference pipeline, a backend API, and an interactive application.