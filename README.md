# *[Title]*

### ✍🏾 Authors: [Author Name 1](https://github.com/), [Author Name 2](https://github.com/), [Author Name 3](https://github.com/), [Author Name 4](https://github.com/), [Author Name 5](https://github.com/)

## Overview

This repository contains the source code and experimental notebooks accompanying the IEEE Access paper on **multi-label competency prediction from course textual descriptions** within the PlaforEDU platform.

The task involves automatically assigning one or more competencies to each course based on its title and description. The study evaluates two complementary paradigms:

- **Supervised multi-label classification**, using TF-IDF, Word2Vec, and BERT textual representations combined with traditional machine learning and AutoML frameworks;
- **Retrieval-Augmented Generation (RAG)**, where candidate competencies are retrieved by semantic similarity and filtered by large language models during inference.

Experiments are conducted on a filtered corpus comprising **342 courses** annotated with **67 competencies**, enabling a controlled comparison between classical classifiers and generative approaches under the same evaluation protocol.

## Repository Structure

- `eda.ipynb`  
  Exploratory data analysis (label frequency, co-occurrence, label cardinality, etc.)

- `llm_proprietary.ipynb`  
  RAG-based labeling using proprietary API models and API embeddings.
  
- `llm_open.ipynb`  
  RAG-based labeling using locally hosted models via Ollama.

- `ml_models.ipynb` *(to be added)*  
  Supervised multi-label baselines + AutoML runs across TF-IDF / Word2Vec / BERT representations.

## Results Summary

The following tables summarize the predictive performance of supervised multi-label classifiers and RAG-based large language models under the same Top-7 evaluation protocol.

Across supervised approaches, contextual BERT embeddings consistently yield higher Partial Hit@7 and Precision@7 values compared to TF-IDF and Word2Vec representations. In contrast, generative models concentrate correct competencies within the top-ranked predictions, with GPT-5-mini achieving the highest Precision@7 among the evaluated LLMs.

### Supervised Models

#### Partial Hit@7 (%)

| Model | TF-IDF | Word2Vec | BERT |
|---------------------------|---------|-----------|--------|
| Random Forest | 79.41 | 72.06 | **83.82** |
| XGBoost | 69.12 | 66.18 | 80.88 |
| MLP (Deep Learning) | 69.12 | 60.29 | 76.47 |
| Gradient Boosting | 75.00 | 66.18 | 73.53 |
| FLAML (AutoML) | 55.88 | 47.06 | 69.12 |
| AutoGluon (AutoML) | 61.76 | 30.88 | 72.06 |

#### Precision@7 (%)

| Model | TF-IDF | Word2Vec | BERT |
|---------------------------|---------|-----------|--------|
| Random Forest | 19.54 | 15.76 | **22.06** |
| XGBoost | 15.34 | 15.76 | 20.59 |
| MLP (Deep Learning) | 15.97 | 13.87 | 20.59 |
| Gradient Boosting | 16.60 | 13.66 | 16.18 |
| FLAML (AutoML) | 0.12 | 0.12 | 0.15 |
| AutoGluon (AutoML) | 0.13 | 0.06 | 0.18 |

### RAG + LLMs

#### Precision@7 (%) and Partial Hit@7 (%)

| Model | Precision@7 (%) | Partial Hit@7 (%) |
|-------------------|------------------|-------------------|
| Gemma-3-27B | 14.36 | 63.45 |
| DeepSeek-R1-70B | 16.59 | 37.43 |
| GPT-4.1-mini | 19.76 | **65.79** |
| GPT-5-mini | **26.11** | 55.26 |