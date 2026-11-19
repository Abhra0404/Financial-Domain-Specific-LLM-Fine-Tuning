# Research and Implementation Plan

## Financial Domain Adaptation and Fine-Tuning of Large Language Models

**Project Type:** Applied Machine Learning / LLM Research

**Primary Focus:** Domain-Adaptive Pretraining, Supervised Fine-Tuning, Parameter-Efficient Fine-Tuning, and Financial NLP Evaluation.

---

# 1. Project Vision

The objective is to investigate how adapting general-purpose Large Language Models to financial-domain data affects their performance on specialized financial NLP tasks.

The project will follow a research-driven implementation process, beginning with understanding existing literature and datasets, followed by baseline evaluation, model adaptation, controlled experimentation, and comparative analysis.

The final outcome should demonstrate both theoretical understanding and practical implementation of modern LLM fine-tuning techniques.

# 2. Research Questions

The project will investigate the following questions.

### RQ1 — Domain Adaptation

Does continued pretraining on financial-domain text improve a general-purpose LLM's ability to understand financial terminology and documents?

### RQ2 — Fine-Tuning Strategies

How do LoRA and QLoRA compare in terms of downstream performance, computational cost, and memory consumption?

### RQ3 — Sequential Adaptation

Does performing Domain-Adaptive Pretraining before Supervised Fine-Tuning provide improvements over direct Supervised Fine-Tuning?

### RQ4 — Task Generalization

Do improvements obtained through financial fine-tuning generalize across sentiment analysis, financial question answering, information extraction, and numerical reasoning?

### RQ5 — Computational Efficiency

What trade-offs exist between model quality, training time, memory consumption, and inference latency?

# 3. Research Hypotheses

The following hypotheses will guide experimentation.

- H1: Exposure to financial-domain text can improve domain-specific language modeling performance.
- H2: Financial SFT can improve downstream performance on financial instruction-following tasks.
- H3: QLoRA can achieve competitive downstream performance with lower memory requirements than conventional LoRA configurations.
- H4: DAPT followed by SFT may outperform direct SFT on selected financial tasks.
- H5: Improvements may vary across tasks, particularly between linguistic understanding and numerical reasoning.

These hypotheses must be tested empirically.

# 4. Phase-Wise Execution Plan

## Phase 1 — Literature Review and Conceptual Understanding

**Objective:** Understand the theoretical foundation of LLM adaptation and financial NLP.

### Topics to Study

#### A. Transformer Architecture
- Transformer architecture.
- Self-attention mechanism.
- Multi-head attention.
- Positional embeddings.
- Tokenization.
- Causal language modeling.

#### B. LLM Pretraining
- Autoregressive language modeling.
- Next-token prediction.
- Pretraining datasets.
- Training objectives.
- Scaling and model capacity.

#### C. Domain Adaptation
- Domain-Adaptive Pretraining.
- Continued pretraining.
- Task-Adaptive Pretraining.
- Domain shift.
- Catastrophic forgetting.

#### D. Fine-Tuning
- Full-parameter fine-tuning.
- Supervised Fine-Tuning.
- Instruction tuning.
- Parameter-efficient fine-tuning.

#### E. LoRA and QLoRA
- Low-rank matrix decomposition.
- Rank and scaling factor.
- Adapter weights.
- Quantization.
- NF4 quantization.
- Double quantization.
- Memory optimization.

### Expected Deliverables

- Literature review notes.
- Technical notes on DAPT and SFT.
- LoRA and QLoRA implementation understanding.
- Initial research bibliography.

---

## Phase 2 — Dataset Research and Acquisition

**Objective:** Identify and prepare appropriate financial datasets.

### Dataset Investigation

#### 1. Financial PhraseBank
Purpose: Financial sentiment classification.

#### 2. FiQA
Purpose: Financial sentiment and question answering.

#### 3. FinQA
Purpose: Numerical reasoning over financial reports.

#### 4. ConvFinQA
Purpose: Multi-turn financial question answering.

#### 5. SEC EDGAR
Purpose: Building a financial-domain corpus for continued pretraining.

### Tasks

1. Inspect dataset structures.
2. Understand dataset licenses and provenance.
3. Identify available labels and annotations.
4. Analyze class distributions.
5. Identify duplicate and overlapping examples.
6. Examine missing or malformed records.
7. Standardize dataset schemas.
8. Create reproducible preprocessing scripts.

### Data Separation

Maintain independent:

- Training dataset.
- Validation dataset.
- Test dataset.

Where possible, use chronological document-level splits to reduce temporal leakage.

Avoid putting multiple excerpts from the same source document into different splits.

### Expected Deliverables

- Dataset inventory.
- Dataset documentation.
- Raw data acquisition scripts.
- Preprocessing pipeline.
- Data quality report.
- Train/validation/test splits.

---

## Phase 3 — Baseline Model Selection and Evaluation

**Objective:** Establish the performance of an unmodified general-purpose model.

### Primary Research Model

Qwen2.5-3B Base.

Model: https://huggingface.co/Qwen/Qwen2.5-3B

### Local Development Model

Qwen2.5-1.5B-Instruct.

Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

### Tasks

1. Understand model architecture and tokenizer.
2. Load the model locally or through cloud infrastructure.
3. Verify inference.
4. Prepare task-specific evaluation prompts.
5. Evaluate the unmodified model.
6. Save predictions and metrics.
7. Document baseline limitations.

### Evaluation

- Sentiment classification accuracy and F1.
- Financial QA Exact Match and token F1.
- Numerical reasoning accuracy.
- Information extraction F1.
- Summarization evaluation.

Use task-appropriate prompts and consistent output parsing.

### Expected Deliverables

- Baseline inference implementation.
- Baseline evaluation scripts.
- Baseline results.
- Model selection rationale.

---

## Phase 4 — Domain-Adaptive Pretraining

**Objective:** Investigate whether exposure to financial-domain language improves the model's domain understanding.

### Data Sources

- Annual financial reports.
- SEC filings.
- Financial statements.
- Earnings call transcripts.
- Other appropriately licensed financial documents.

### Preprocessing

1. Extract text.
2. Remove irrelevant formatting.
3. Normalize whitespace.
4. Remove duplicates.
5. Filter low-quality documents.
6. Preserve document metadata.
7. Tokenize using the selected model tokenizer.
8. Prepare causal language modeling sequences.

### Training Objective

Continue training the pretrained model using the causal language modeling objective.

The model learns to predict subsequent tokens in financial-domain text.

### Initial Training Configuration

- Base model: Qwen2.5-3B.
- Sequence length: Initially 1024, increasing if resources permit.
- Precision: BF16 where supported.
- Learning rate: Small, validated through pilot experiments.
- Training: Controlled continuation of language-model pretraining.

Investigate full-parameter continued pretraining versus parameter-efficient continued adaptation depending on compute availability.

### Evaluation

- Financial-domain validation loss.
- General-domain validation loss.
- Downstream financial benchmark performance.
- Signs of catastrophic forgetting.

### Expected Deliverables

- Curated financial corpus.
- DAPT training pipeline.
- Adapted model checkpoint or adapter.
- Training logs.
- Evaluation report.

---

## Phase 5 — Supervised Fine-Tuning

**Objective:** Teach the model to perform specialized financial tasks.

### Training Tasks

1. Financial question answering.
2. Financial sentiment classification.
3. Structured financial information extraction.
4. Financial document summarization.

### Dataset Formatting

Convert source datasets into consistent instruction-response formats.

Each example should contain:

- Instruction.
- Input context.
- Expected response.

For structured extraction, enforce a documented JSON schema.

For numerical reasoning, preserve the supporting evidence and expected numerical result.

### Initial Training Strategy

Implement SFT using Hugging Face Transformers and TRL.

Begin with LoRA.

Then investigate QLoRA using a 4-bit quantized base model.

### Initial LoRA Configuration

| Parameter | Initial Setting |
|---|---|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Learning Rate | 2e-4 |
| Sequence Length | 2048 |
| Micro Batch Size | 1 |
| Gradient Accumulation | 16 |
| Epochs | 2–3 |
| Scheduler | Cosine |

These settings are starting points. Validate them through pilot training runs.

### Expected Deliverables

- Financial instruction dataset.
- SFT training scripts.
- LoRA adapter.
- QLoRA adapter.
- Training configurations.
- Experiment logs.

---

## Phase 6 — Comparative Experiments

**Objective:** Compare adaptation strategies under controlled conditions.

### Experiment Matrix

| ID | Model Configuration |
|---|---|
| E0 | Original pretrained model |
| E1 | Original model + SFT with LoRA |
| E2 | Original model + SFT with QLoRA |
| E3 | DAPT + SFT |
| E4 | Comparative and ablation experiments |

Use the same primary model family and evaluation datasets wherever possible.

Control training data, data splits, evaluation prompts, and random seeds.

### Comparisons

- Performance differences.
- Training duration.
- GPU memory usage.
- Trainable parameter count.
- Model size.
- Inference latency.

### Expected Deliverables

- Experiment tracking dashboard.
- Comparative benchmark tables.
- Performance plots.
- Ablation study.
- Experiment analysis report.

---

## Phase 7 — Model Evaluation and Error Analysis

**Objective:** Determine whether the adapted models provide meaningful improvements.

### Metrics

| Task | Metrics |
|---|---|
| Sentiment | Accuracy, Precision, Recall, F1 |
| Financial QA | Exact Match, Token F1 |
| Numerical Reasoning | Numerical Accuracy, Execution Accuracy |
| Information Extraction | Precision, Recall, F1 |
| Summarization | ROUGE-L, Human Evaluation |

### Error Analysis

Investigate:

- Incorrect financial terminology.
- Hallucinated financial figures.
- Incorrect numerical calculations.
- Missing extracted fields.
- Incorrect sentiment classification.
- Unsupported statements in summaries.

For numerical reasoning, independently execute generated calculations and verify their answers.

### Generalization

Evaluate the possibility of performance degradation on general-domain tasks after domain adaptation.

Document both improvements and regressions.

### Expected Deliverables

- Evaluation framework.
- Error analysis report.
- Final benchmark results.
- Limitations analysis.

---

## Phase 8 — Inference and Deployment

**Objective:** Make the resulting model accessible through a reproducible inference interface.

### Backend

Use FastAPI.

### Endpoints

- `/predict/sentiment`
- `/predict/qa`
- `/predict/extraction`
- `/predict/summarization`
- `/health`

### Features

- Model loading.
- Input validation.
- Structured JSON responses.
- Error handling.
- Request logging.
- Inference timing.

### Deployment

- Docker containerization.
- Model artifact management.
- API documentation.
- Basic demonstration interface.

The inference service should clearly distinguish model-generated outputs from independently verified calculations.

### Expected Deliverables

- Inference API.
- Docker configuration.
- API documentation.
- Demonstration interface.

---

## Phase 9 — Research Documentation and Final Analysis

**Objective:** Document the methodology, experiments, findings, and limitations.

### Final Documentation

1. Research motivation.
2. Related work.
3. Dataset methodology.
4. Model architecture.
5. Domain-adaptation methodology.
6. Fine-tuning methodology.
7. Experimental setup.
8. Evaluation methodology.
9. Results and comparisons.
10. Error analysis.
11. Limitations.
12. Future work.

### Final Deliverables

- Complete source code.
- Reproducible experiment configurations.
- Dataset documentation.
- Model adapters and artifacts.
- Benchmark results.
- Technical report.
- Final README.

---

# 5. Hardware and Compute Strategy

## Local Machine

Apple MacBook M4, 16 GB unified memory, 512 GB storage.

Use for:

- Data exploration.
- Dataset preprocessing.
- Tokenization.
- Small-model inference.
- Lightweight MLX-LM experiments.
- Evaluation development.
- API development.

## Cloud Infrastructure

Preferred: NVIDIA A100 40 GB.

Alternative: NVIDIA L4 24 GB.

Use for:

- Primary 3B model experiments.
- DAPT.
- SFT.
- LoRA and QLoRA comparisons.
- Larger evaluation workloads.

Start with short pilot runs before allocating substantial GPU time.

---

# 6. Suggested Timeline

The project should be implemented incrementally.

| Phase | Estimated Duration |
|---|---|
| Literature review | 1 week |
| Dataset investigation | 1 week |
| Baseline evaluation | 3–4 days |
| DAPT experiments | 1–2 weeks |
| SFT implementation | 1 week |
| Comparative experiments | 1 week |
| Evaluation and analysis | 1 week |
| API and deployment | 3–4 days |
| Documentation | 3–4 days |

**Estimated total: 7–9 weeks**, depending on dataset preparation, compute availability, and experiment duration.

The timeline is flexible. Research quality takes priority over completing every experiment quickly.

---

# 7. Success Criteria

The project will be considered technically complete when:

- A reproducible dataset pipeline exists.
- A pretrained baseline has been evaluated.
- At least one financial SFT model has been trained.
- LoRA and QLoRA have been investigated.
- DAPT has been implemented or its feasibility documented through controlled experiments.
- Adapted models have been evaluated on held-out benchmarks.
- Computational efficiency has been measured.
- Results are reproducible and documented.
- A working inference API is available.

A positive performance improvement is not a prerequisite for scientific success. A carefully controlled negative result is also a valid research finding.

---

# 8. Development Principles

1. Understand each technique before implementing it.
2. Avoid blindly copying fine-tuning tutorials.
3. Keep data preparation separate from model training.
4. Maintain reproducible experiment configurations.
5. Never evaluate on training data.
6. Report actual results without exaggeration.
7. Track compute consumption.
8. Document failed experiments and technical decisions.
9. Keep model checkpoints and datasets outside Git.
10. Prioritize experimental rigor over the number of features.

---

# 9. Immediate Next Steps

Before beginning model training:

- [ ] Create the GitHub repository.
- [ ] Set up the local Python environment.
- [ ] Study Transformer architecture and causal language modeling.
- [ ] Study DAPT, SFT, LoRA, and QLoRA.
- [ ] Investigate the selected financial datasets.
- [ ] Finalize the primary research model.
- [ ] Prepare the baseline evaluation methodology.

**Current Status:** Planning and research preparation.

**Next Milestone:** Complete the literature review and dataset investigation before implementing the baseline evaluation.