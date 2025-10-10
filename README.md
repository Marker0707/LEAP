# LEAP: LLM-Enhanced Automated Phenotyping
![](https://img.shields.io/badge/python-3.12+-blue.svg)  ![](https://img.shields.io/badge/license-MIT-green.svg)  ![](https://img.shields.io/badge/version-2.0.0-orange.svg)

LEAP (LLM-Enhanced Automated Phenotyping) is a Python package that combines Large Language Models (LLMs) with semantic similarity analysis to automatically extract and map phenotypic descriptions from medical texts to Human Phenotype Ontology (HPO) terms. It is designed for researchers and clinicians who need an end-to-end pipeline from raw clinical narratives to structured phenotype and gene-ranking outputs.

## 🆕 What's New in V2
- **Simplified Configuration**: All configuration now lives in the `LEAP` constructor—no external `config.py` required.
- **Enhanced Logging**: Centralised logging utilities make it easier to trace each step of the pipeline.
- **Reranking Support**: Optional cross-encoder reranking improves HPO term selection accuracy.
- **Better Error Handling**: Built-in validation for paths and network calls.
- **Improved Modularity**: Components are cleanly separated for easier extension.

## 🌟 Features
- **Multi-LLM Support**: Works with OpenAI GPT-4o and local Ollama models (Gemma3, DeepSeek, Qwen, Llama).
- **Automated Phenotype Extraction**: Uses LLMs to turn clinical narratives into medical-grade phenotype phrases.
- **HPO Mapping**: Maps extracted phenotypes to standardized HPO terms via SentenceTransformers.
- **Cross-Encoder Reranking**: Optional reranking step to refine HPO matches.
- **Gene Ranking**: Integrates with PhenoApt to prioritise candidate genes.
- **Caching & Reuse**: Automatically caches embeddings for fast subsequent runs.

## 🧭 Workflow Overview
- Ingest unstructured clinical text.
- Use an LLM client to extract phenotype statements.
- Embed statements and compute semantic similarity against the cached HPO matrix.
- Optionally rerank and weight results to retain the most specific HPO identifiers.
- (Optional) Submit HPO terms to a gene ranking service and inspect structured outputs.

## 🚀 Installation

### Requirements
- Python >= 3.12
- `pip` with the ability to install Python wheels (PyTorch is pulled in through `sentence-transformers`)
- **Optional**: [Ollama](https://ollama.com/) >= 0.11.3 for running local Gemma/Qwen/DeepSeek/Llama models
- Internet access the first time you build the HPO embedding matrix (SentenceTransformer downloads model weights)

### (Optional) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install from source
```bash
git clone https://github.com/marker0707/LEAP.git
cd LEAP
pip install -e .
```

### Verify the installation
```bash
python - <<'PY'
import leap
print("LEAP version:", leap.__version__)
PY
```

### Download the Human Phenotype Ontology
Download the latest `hp.obo` from the [HPO portal](https://hpo.jax.org/data/ontology) and place it where your code can read it.

You can optionally prepare an add-on CSV (`hp_addon_path`) with extra synonyms. The file should provide two columns: `HP_ID` and `info`.

## 🛠️ Usage Guide

### 1. Configure an LLM client
Pick or implement an `LLMClient` for phenotype extraction. The package ships with clients for OpenAI and several Ollama-hosted models.
```python
from leap.llm_client import OpenAIClient

llm_client = OpenAIClient(api_key="your-openai-api-key")
```
For local deployment you can swap in `Gemma3Client`, `DeepSeekClient`, `QwenClient`, or `LlamaClient` by providing the correct Ollama endpoint and model size.

### 2. Instantiate `LEAP`
```python
from leap import LEAP

leap = LEAP(
    model="all-MiniLM-L6-v2",
    llm_client=llm_client,
    hp_obo_path="./hp.obo",
    save_path="~/.cache/leap-hpo",   # optional: customise the cache directory
    hp_addon_path=""                 # optional: CSV with additional HPO text
)
```
On the first run, LEAP will build `hpo_embeddings_matrix.pkl` in the save path; subsequent runs reuse the cached matrix for faster start-up.

#### Constructor arguments
| Argument | Required | Description |
| --- | --- | --- |
| `model` | ✅ | SentenceTransformer model name or local path used to embed phenotype text. |
| `llm_client` | ✅ | Instance implementing `LLMClient.extract_phenotypes`. Determines how phenotypes are generated. |
| `hp_obo_path` | ✅ | Path to the `hp.obo` ontology file. Needed to build and validate the HPO knowledge base. |
| `save_path` | Optional | Directory where embedding matrices and graph caches are stored. Defaults to `~/.cache/leap-hpo`. Must be writable. |
| `hp_addon_path` | Optional | CSV of extra HPO text snippets (`HP_ID`, `info`) appended during embedding. Useful for domain-specific synonyms. |

### 3. Extract phenotypes from clinical text
```python
medical_text = """
Patient presents with intellectual disability, seizures,
and distinctive facial features including a broad forehead
and widely spaced eyes.
"""

result = leap.convert_ehr(
    content=medical_text,
    rerank=False,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",  # required if rerank=True
    furthest=True,
    use_weighting=False,
    top_k=500,
    retrieve_cutoff=0.7,
    rerank_cutoff=0.5
)
```

#### Method parameters
| Parameter | Description |
| --- | --- |
| `content` | Raw clinical narrative passed to the LLM client. |
| `rerank_model_name` | Cross-encoder identifier or path. Mandatory when `rerank=True`; ignored otherwise. |
| `rerank` | Enables cross-encoder reranking for higher precision. |
| `furthest` | Keeps the most specific HPO terms by removing ancestors from the final list. |
| `use_weighting` | Applies branch-aware weighting to emphasise consistent ontology branches. |
| `top_k` | Number of nearest neighbours to pull from the embedding matrix during retrieval. |
| `retrieve_cutoff` | Similarity threshold for the initial retrieval stage. |
| `rerank_cutoff` | Score threshold after reranking (only applied when `rerank=True`). |

### 4. Inspect results
`convert_ehr` returns a `LEAPResult` object capturing every stage of the pipeline:
```python
print("Final HPO IDs:", result.final_leap_result)
print("LLM phrases:", result.llm_result)
print("Top retrieval candidates:\n", result.retrieve_result.head())

if result.weighted_result is not None:
    print("Weighted selection:\n", result.weighted_result.head())

if result.rerank_result is not None:
    print("Rerank summary:\n", result.rerank_result.head())
```

- **`final_leap_result`**: Deduplicated, optionally branch-pruned HPO IDs to use downstream.
- **`llm_result`**: Raw phenotype phrases returned by the LLM client.
- **`retrieve_result`**: Pandas DataFrame of top retrieval candidates for each phrase.
- **`weighted_result`**: Optional DataFrame when `use_weighting=True`.
- **`rerank_result`**: Optional DataFrame containing cross-encoder scores when reranking is enabled.

Each `LEAPResult` also records the embedding model, LLM client name, and creation timestamp for reproducibility.

### 5. Optional: Rank genes with PhenoApt
You can call the gene-ranking step on demand from the result object. The request hits the public PhenoApt API and returns a structured DataFrame.
```python
result.rank_gene(tool="PhenoApt", weight_list=None)

if result.gene_rank_result:
    phenoapt = result.gene_rank_result[0]
    print("Gene ranking status:", phenoapt.response_status)
    print(phenoapt.response_df[["gene_symbol", "score"]].head())
```
Provide `weight_list` if you want phenotype-specific weights; it must be the same length as `final_leap_result`.

## 🤖 Supported LLM Clients
| Client | Typical Usage | Notes |
| --- | --- | --- |
| `OpenAIClient` | Hosted GPT-4o models | Requires an OpenAI API key and internet access. |
| `Gemma3Client` | Local Ollama models (`gemma3`) | Provide `endpoint` and `size` (`4b`, `12b`, `27b`). |
| `DeepSeekClient` | Local DeepSeek-R1 via Ollama | Supports `1.5b`, `7b`, `14b`, `32b`; strips `<think>` output automatically. |
| `QwenClient` | Local Qwen3 via Ollama | Choose a supported model size (`0.6b`–`235b`). |
| `LlamaClient` | Local Llama 3.x via Ollama | Default `llama3.3:70b`; override via the `model` argument. |

To integrate a different provider, subclass `LLMClient` and implement the `extract_phenotypes` method returning a list of strings.

## 🔧 Advanced Configuration

### Cross-Encoder reranking
```python
result = leap.convert_ehr(
    content=medical_text,
    rerank=True,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    furthest=True,
    use_weighting=False,
    retrieve_cutoff=0.7,
    rerank_cutoff=0.1
)
```
The reranking step reorders retrieval candidates with a cross-encoder and applies the `rerank_cutoff` threshold before generating the final HPO list.

### Weighted phenotype mapping
```python
result = leap.convert_ehr(
    content=medical_text,
    rerank=False,
    use_weighting=True,
    retrieve_cutoff=0.7
)
```
Weighting analyses the top ontology branch for each phrase and promotes the most informative node when multiple terms belong to the same branch.

### Custom gene-ranking weights
```python
weights = [1.0, 0.8, 1.2]
result.rank_gene(tool="PhenoApt", weight_list=weights)
```
The weights must align with the order of `result.final_leap_result`.

### Managing the embeddings cache
- `save_path` controls where `hpo_embeddings_matrix.pkl`, `database_info.json`, and NetworkX graph caches are stored.
- The cache encodes which SentenceTransformer model and HPO file were used. Changing either triggers a rebuild.
- Share the same cache directory across runs or machines to avoid recomputation.

### Extending with custom LLM clients
```python
from leap.llm_client import LLMClient

class MyCustomClient(LLMClient):
    def extract_phenotypes(self, content: str) -> list[str]:
        # Call your favourite API and return a list of phenotype strings
        return call_my_service(content)
```
Pass any `LLMClient` subclass into `LEAP` to plug in new providers or prompt strategies.

## 🧪 Example Script
See `example_usage.py` for a fully worked example that wires an Ollama client, runs `convert_ehr`, and prints the resulting HPO IDs.

## 🧰 Logging & Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

leap = LEAP(
    model="all-MiniLM-L6-v2",
    llm_client=llm_client,
    hp_obo_path="./hp.obo"
)
```
Setting the global log level to `DEBUG` enables detailed tracing from matrix validation through retrieval and reranking. Logs use a consistent formatter via `leap.logging_utils`.

## 📄 License
This project is licensed under the MIT License.
