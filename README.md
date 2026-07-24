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

The source dataset contains **342 courses** and **67 competencies**. After preprocessing and rare-label filtering, the experimental pipeline uses **338 courses** and **53 competencies**, split into **270 training courses** and **68 test courses** under a unified Top-k evaluation protocol.

## Repository Structure

- `eda.ipynb`  
  Exploratory data analysis (label frequency, co-occurrence, label cardinality, etc.)

- `processing.ipynb`
  ETL/pre-processing step that reads `dataset.csv`, aggregates course labels, filters rare competencies, and exports `processed.csv`.

- `data_segregation.ipynb`
  Train/test segregation step that reads `processed.csv` and exports `train.csv` and `test.csv`.

- `ml_models.ipynb`  
  Supervised multi-label baselines using TF-IDF, Word2Vec, and BERT-based representations. This notebook reads `train.csv` and `test.csv`.

- `autoML.ipynb`  
  Automated machine learning experiments for model selection and hyperparameter optimization. This notebook reads `train.csv` and `test.csv`.

- `autokeras.ipynb`  
  AutoKeras-based experiments for automated neural architecture search. This notebook reads `train.csv` and `test.csv`.

- `llm_proprietary.ipynb`  
  RAG-based labeling using OpenAI API models and API embeddings. The exported experiments evaluate `gpt-4.1-mini` and `gpt-5-mini` with `text-embedding-3-small` on the same `test.csv` courses used by the supervised experiments.

- `llm_open.ipynb`  
  RAG-based labeling using locally hosted models via Ollama. The exported experiments evaluate `gemma3:27b` and `deepseek-r1:70b` with `nomic-embed-text`, assuming local execution on an NVIDIA RTX 6000 Ada 48GB GPU.

- `dataset.csv`  
  Source dataset containing course descriptions and associated competencies before the preprocessing and rare-label filtering steps.

- `processed.csv`, `train.csv`, `test.csv`
  Generated pipeline artifacts used by the supervised and RAG/LLM notebooks.

## Results Summary

The paper reports best predictive results across the evaluated Top-k cutoffs, with `k` in `{1, 3, 5, 7, 10}`. Values below follow the paper's `[0, 1]` scale and indicate the cutoff at which each best value was achieved. The exported metrics, sensitivity tables, timings, predictions, and configurations are stored under `results/`.

### Exploratory Findings

Before rare-label filtering, the corpus contains **342 courses** annotated with **67 distinct competencies**. The competency distribution is long-tailed: the most frequent competency appears in **107 courses**, the median frequency is **10**, and the first and third quartiles are **5** and **16**. After filtering competencies that appear in fewer than five courses, the common experimental label space is reduced to **53 competencies** and **338 courses**.

The co-occurrence analysis identifies **717 distinct pairs** of competencies. The most frequent pair, `(Ensino, Orientação)`, appears in **71 courses** and represents approximately **4.3%** of observed co-occurrences. Most courses contain between **1 and 3 competencies**, with a median label cardinality of **2**.

### Supervised Models

Among manually configured supervised models, BERT-based representations produced the strongest configurations overall. The main paper-level highlights are:

| Result | Best configuration | Value |
|---|---|---:|
| Micro-F1 | Random Forest + BERT | 0.4074 (`k=3`) |
| Macro-F1 | Binary Relevance (LR) + BERT | 0.2522 (`k=3`) |
| Hamming Loss | Random Forest + BERT; MLP + BERT | 0.0483 (`k=1`) |
| Subset Accuracy | Binary Relevance (LR) + TF-IDF; MLP + BERT | 0.0882 (`k=1`) |
| Precision@k | Random Forest + BERT; MLP + BERT | 0.5000 (`k=1`) |
| Recall@k | Binary Relevance (LR) + BERT | 0.7370 (`k=10`) |
| Partial Hit@k | Binary Relevance (LR) + BERT; Random Forest + BERT | 0.8824 (`k=10`) |

Training and inference costs varied substantially. Binary Relevance with Word2Vec had the lowest training time (**0.078 s**), while Gradient Boosting with BERT required **258.180 s**. MLP with Word2Vec achieved the lowest inference latency (**0.012 ms/sample**). Random Forest with BERT required **20.010 s** for training and achieved the highest Micro-F1.

### AutoML Models

The automated machine learning results varied across frameworks and textual representations. FLAML with TF-IDF achieved the strongest AutoML values for most metrics:

| Result | Best AutoML configuration | Value |
|---|---|---:|
| Micro-F1 | FLAML + TF-IDF | 0.3439 (`k=3`) |
| Macro-F1 | FLAML + BERT | 0.2339 (`k=3`) |
| Hamming Loss | FLAML + TF-IDF | 0.0511 (`k=1`) |
| Subset Accuracy | FLAML + TF-IDF | 0.0882 (`k=1`) |
| Precision@k | FLAML + TF-IDF | 0.4265 (`k=1`) |
| Recall@k | FLAML + TF-IDF | 0.5727 (`k=10`) |
| Partial Hit@k | FLAML + TF-IDF | 0.7500 (`k=10`) |

AutoKeras was evaluated from raw textual input. Its best Micro-F1 and Precision@k were both obtained at `k=1`, with **0.1818** and **0.3235**, respectively. Its best Recall@k and Partial Hit@k occurred at `k=10`, with **0.2891** and **0.5294**.

### RAG + LLMs

The RAG branch evaluates four generative configurations and non-generative baselines. GPT-4.1-mini achieved the strongest Micro-F1, Precision@k, and Partial Hit@k among the LLMs, while GPT-5-mini achieved the strongest Macro-F1 and Recall@k:

| Result | Best RAG/LLM configuration | Value |
|---|---|---:|
| Micro-F1 | GPT-4.1-mini | 0.3100 (`k=5`) |
| Macro-F1 | GPT-5-mini | 0.2382 (`k=5`) |
| Hamming Loss | GPT-4.1-mini | 0.0519 (`k=1`) |
| Subset Accuracy | Gemma 3 27B; DeepSeek-R1 70B | 0.0441 (`k=1`) |
| Precision@k | GPT-4.1-mini | 0.3971 (`k=1`) |
| Recall@k | GPT-5-mini | 0.3668 (`k=10`) |
| Partial Hit@k | GPT-4.1-mini | 0.6176 (`k=5,7,10`) |

The retrieval-only baseline with `text-embedding-3-small` achieved the highest Recall@k (**0.5659**, `k=10`) and Partial Hit@k (**0.7794**, `k=10`) among RAG-related configurations, while the strongest LLM configuration achieved higher Micro-F1, Macro-F1, and Precision@k. The frequency baseline reached Recall@k of **0.4998** and Partial Hit@k of **0.7059** at `k=10`.

### Runtime Comparison

The frequency baseline had negligible runtime. Retrieval-only inference required **0.39 s/course** with `text-embedding-3-small` and **0.04 s/course** with `nomic-embed-text`. Among generative RAG configurations, the lowest runtime was obtained by GPT-4.1-mini:

| Method | Average runtime per course |
|---|---:|
| GPT-4.1-mini | 2.95 s |
| Gemma 3 27B | 10.99 s |
| GPT-5-mini | 23.62 s |
| DeepSeek-R1 70B | 288.39 s |

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

Run the pipeline in this order for supervised experiments:

``` bash
eda.ipynb
processing.ipynb
data_segregation.ipynb
ml_models.ipynb / autoML.ipynb / autokeras.ipynb
llm_open.ipynb / llm_proprietary.ipynb
```

The supervised and RAG/LLM notebooks export updated metrics, sensitivity-by-k tables, timing tables, configuration logs, and predictions under `results/`.

### 6. Running local LLM experiments

The notebook **llm_open.ipynb** uses **Ollama** to run local models.
The open LLM experiments are identified in the exported artifacts as
`ollama_gemma3_27b_nomic_embed_rag` and
`ollama_deepseek_r1_70b_nomic_embed_rag`, while the executable Ollama
model IDs remain `gemma3:27b`, `deepseek-r1:70b`, and
`nomic-embed-text`.

The reported runtime assumes local execution with an **NVIDIA RTX 6000
Ada 48GB** GPU. Running the same notebook on CPU or on a different GPU
can substantially change the timing results.

Install Ollama: https://ollama.com

Download the model and embedding model:

``` bash
ollama pull gemma3:27b
ollama pull deepseek-r1:70b
ollama pull nomic-embed-text
```

Start Ollama before executing the notebook.

The proprietary LLM experiments in **llm_proprietary.ipynb** are
identified as `openai_gpt4.1mini_text_embedding_3_small_rag` and
`openai_gpt5mini_text_embedding_3_small_rag`. They use OpenAI managed
infrastructure through the API, so their timing should be interpreted as
API/runtime latency rather than local GPU runtime.
