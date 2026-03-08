# A Framework for Automated Multi-Label Competency Mapping from Course Descriptions with Supervised Learning and Retrieval-Augmented Large Language Models

### ✍🏾 Authors: 
[Thaís Medeiros](https://github.com/thaisaraujom), 
[Morsinaldo Medeiros](https://github.com/morsinaldo), 
[Matheus Andrade](https://github.com/DinizMaths), 
[Fabiane B. da Silva](https://github.com/fabibeletti), 
[Silvan Ferreira](https://github.com/silvaan), 
[Raymundo C. M. Ferreira Filho](https://github.com/pakafe), 
[Otávio A. de L. Júnior](https://scholar.google.com/citations?user=93G_lY0AAAAJ&hl=en&oi=ao),
[Thiago M. Barros](https://github.com/tmedeirosb), 
[Ivanovitch Silva](https://github.com/ivanovitchm)

## Overview

This repository contains the source code and experimental notebooks accompanying the IEEE Access paper **"A Framework for Automated Multi-Label Competency Mapping from Course Descriptions with Supervised Learning and Retrieval-Augmented Large Language Models"**, which introduces a framework for automated competency mapping from course textual descriptions within the PlaforEDU platform.

The framework supports the automatic assignment of one or more competencies to each course based on its title and description. The study evaluates two complementary paradigms:

- **Supervised multi-label classification**, using TF-IDF, Word2Vec, and BERT textual representations combined with traditional machine learning and AutoML frameworks;
- **Retrieval-Augmented Generation (RAG)**, where candidate competencies are retrieved by semantic similarity and filtered by large language models during inference.

Experiments are conducted on a filtered corpus comprising **342 courses** annotated with **67 competencies**, enabling a controlled comparison between classical classifiers and generative approaches under a unified Top-k evaluation protocol.

## Repository Structure

- `eda.ipynb`  
  Exploratory data analysis (label frequency, co-occurrence, label cardinality, etc.)

- `ml_models.ipynb`  
  Supervised multi-label baselines using TF-IDF, Word2Vec, and BERT-based representations.

- `autoML.ipynb`  
  Automated machine learning experiments for model selection and hyperparameter optimization.

- `autokeras.ipynb`  
  AutoKeras-based experiments for automated neural architecture search.

- `llm_proprietary.ipynb`  
  RAG-based labeling using proprietary API models and API embeddings.

- `llm_open.ipynb`  
  RAG-based labeling using locally hosted models via Ollama.

- `dataset.csv`  
  Filtered dataset containing course descriptions and associated competencies used in the experiments.

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

Segue o conteúdo em Markdown para colocar em uma célula de notebook (Markdown cell):

## How to Run

The experiments in this repository were implemented using **Python
3.11.11** and executed through Jupyter notebooks. The recommended
environment manager is **Miniconda/Conda**, and the notebooks can be
executed directly in **Visual Studio Code (VS Code)** using the Python
and Jupyter extensions.

### 1. Clone the repository

``` bash
git clone https://github.com/plaforedu-ifrn/IEEEAccess-PlaforEDU.git
cd IEEEAccess-PlaforEDU
```

### 2. Create a Conda environment

Create and activate a Python environment using Conda:

``` bash
conda create -n plaforedu python=3.11
conda activate plaforedu
```

### 3. Install dependencies

Install all required Python libraries:

``` bash
pip install -r requirements.txt
```

### Environment variables for proprietary models (.env)

Some notebooks (e.g., **llm_proprietary.ipynb**) use proprietary LLM
APIs. To run these experiments, create a `.env` file in the root
directory of the repository and define the required API keys.

Example:

``` bash
OPENAI_API_KEY=your_api_key_here
```

or other provider credentials depending on the model used.

Make sure the `.env` file is loaded before executing the notebooks
(e.g., using `python-dotenv`).

### 4. Open the notebooks

Open the repository folder in **Visual Studio Code**.

Make sure the following extensions are installed:

-   **Python**
-   **Jupyter**

Then select the **plaforedu** Conda environment as the notebook kernel
before running the cells.

### 5. Execute the notebooks

The notebooks can be executed independently.

### 6. Running local LLM experiments

The notebook **llm_open.ipynb** uses **Ollama** to run local models.

Install Ollama: https://ollama.com

Example model download:

``` bash
ollama pull gemma3:27b
```

Start Ollama before executing the notebook.

## Dataset

The file **dataset.csv** contains the filtered dataset used in the
experiments, consisting of **342 courses annotated with 67
competencies**.
