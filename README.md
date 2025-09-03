# LEAP: LLM-Enhanced Automated Phenotyping
![](https://img.shields.io/badge/python-3.12+-blue.svg)  ![](https://img.shields.io/badge/license-MIT-green.svg)  ![](https://img.shields.io/badge/version-2.0.0-orange.svg)

LEAP (LLM-Enhanced Automated Phenotyping) is a Python package that combines Large Language Models (LLMs) with semantic similarity analysis to automatically extract and map phenotypic descriptions from medical texts to Human Phenotype Ontology (HPO) terms.

## 🆕 What's New in V2
- **Simplified Configuration**: Removed separate config.py file - configuration is now handled directly in the LEAP class constructor
- **Enhanced Logging**: Comprehensive logging system for better debugging and monitoring
- **Reranking Support**: Added cross-encoder reranking capabilities for improved HPO term mapping accuracy
- **Better Error Handling**: Robust path validation and error handling mechanisms
- **Improved Modularity**: Refactored codebase for better maintainability and code organization

## 🌟 Features
+ **Multi-LLM Support**: Compatible with OpenAI GPT-4, Google Gemma3, DeepSeek, Qwen, and Llama models
+ **Automated Phenotype Extraction**: Extract phenotypic descriptions from clinical texts using advanced LLMs
+ **HPO Mapping**: Map extracted phenotypes to standardized HPO (Human Phenotype Ontology) terms using sentence transformers
+ **Cross-Encoder Reranking**: Improve mapping accuracy with optional reranking using cross-encoder models
+ **Gene Ranking**: Prioritize candidate genes based on phenotype profiles using PhenoApt
+ **Enhanced Logging**: Comprehensive logging system for debugging and monitoring
+ **Flexible Configuration**: Direct configuration through class constructor without external config files

## 🚀 Installation
### Requirements
+ Python >= 3.12.0
+ Ollama >= 0.11.3

### Install from Source
```bash
git clone https://github.com/marker0707/LEAP.git
cd LEAP
pip install -e .
```

### Download hp.obo
```bash
https://hpo.jax.org/data/ontology
```

## 🛠️ Quick Start

### Basic Usage
```python
from leap import LEAP
from leap.llm_client import OpenAIClient

# Initialize LLM client
llm_client = OpenAIClient(api_key="your-openai-api-key")

# Initialize LEAP (V2 - Direct Configuration)
leap = LEAP(
    model="all-MiniLM-L6-v2",  # Sentence transformer model
    llm_client=llm_client,
    hp_obo_path="./hp.obo",    # Path to HPO ontology file
    save_path="~/.cache/leap-hpo"  # Optional: custom save path for embeddings
)

# Process medical text
medical_text = """
Patient presents with intellectual disability, seizures, 
and distinctive facial features including a broad forehead 
and widely spaced eyes.
"""

# Extract phenotypes and map to HPO
result = leap.convert_ehr(
    content=medical_text,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    rerank=False,           # Enable reranking for better accuracy
    furthest=True,          # Retain most specific HPO terms
    use_weighting=False,    # Use simple top-1 mapping
    retrieve_cutoff=0.7,    # Similarity threshold for retrieval
    rerank_cutoff=0.5       # Similarity threshold for reranking
)

print("Extracted HPO terms:", result.final_leap_result)
print("LLM-extracted phenotypes:", result.llm_result)
```

### Gene Ranking
```python
# Perform end-to-end analysis with gene ranking
result = leap.ehr2gene(
    content=medical_text,
    furthest=True,
    use_weighting=False,
    tool="PhenoApt"  # Gene ranking tool
)

# Access gene ranking results
if result.gene_rank_result:
    gene_df = result.gene_rank_result[0].response_df
    top_genes = gene_df.head(10)
    print("Top candidate genes:")
    print(top_genes[['gene_symbol', 'score']])
```

### Using Different LLM Clients
```python
from leap.llm_client import Gemma3Client, DeepSeekClient, QwenClient, LlamaClient

# Ollama-based models (requires local Ollama installation)
gemma_client = Gemma3Client(endpoint="http://localhost:11434/api/generate", size="27b")
deepseek_client = DeepSeekClient(endpoint="http://localhost:11434/api/generate", size="32b")
qwen_client = QwenClient(endpoint="http://localhost:11434/api/generate", size="32b")
llama_client = LlamaClient(endpoint="http://localhost:11434/api/generate", model="llama3.3:70b")

# Use with LEAP (V2 configuration)
leap_gemma = LEAP(
    model="all-MiniLM-L6-v2",
    llm_client=gemma_client,
    hp_obo_path="./hp.obo"
)
```

## 📊 Advanced Features

### Cross-Encoder Reranking
Enable advanced reranking to improve HPO term mapping accuracy:

```python
result = leap.convert_ehr(
    content=medical_text,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    rerank=True,            # Enable reranking
    furthest=True,
    use_weighting=False,
    retrieve_cutoff=0.7,    # Initial retrieval threshold
    rerank_cutoff=0.1       # Reranking threshold (lower = more strict)
)
```

### Weighted Phenotype Mapping
Enable advanced weighting to improve HPO term selection:

```python
result = leap.convert_ehr(
    content=medical_text,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    furthest=True,
    use_weighting=True,     # Use top-10 branch analysis for better mapping
    retrieve_cutoff=0.7
)
```

### Custom Gene Ranking Weights
Provide custom weights for different phenotypes:

```python
# Assuming you have HPO terms and want to weight them differently
hpo_terms = ['HP:0001249', 'HP:0001250', 'HP:0000252']
weights = [1.0, 0.8, 1.2]  # Custom importance weights

result = leap.ehr2gene(
    content=medical_text,
    weight_list=weights
)
```

### Configuration Parameters

The LEAP class now accepts configuration directly in the constructor:

- `model`: Sentence transformer model name or path
- `llm_client`: LLM client instance for phenotype extraction
- `hp_obo_path`: Path to the HPO ontology file (hp.obo)
- `save_path`: Optional custom path for saving embeddings (defaults to ~/.cache/leap-hpo)
- `hp_addon_path`: Optional path for additional HPO terms

### Logging

LEAP V2 includes comprehensive logging. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now LEAP will provide detailed logging information
leap = LEAP(model="all-MiniLM-L6-v2", llm_client=llm_client, hp_obo_path="./hp.obo")
```

## 💡 Complete Example

Here's a complete example demonstrating LEAP V2 functionality with a clinical case:

```python
import leap
from leap.llm_client import Gemma3Client

# Clinical case description
content = """
Chief Complaint: Progressive weight gain and excessive eating behavior for the past 2 years.

History of Present Illness: The patient is a 7-year-old boy brought in by his parents 
with concerns of abnormal weight gain despite attempts at dietary restriction. Since around 
the age of 5, he has exhibited persistent food-seeking behavior, including stealing food 
and overeating. Parents also report decreased muscle tone and delayed motor milestones in infancy.

Physical Examination: Obese child with characteristic facial features (almond-shaped eyes, 
narrow bifrontal diameter, downturned mouth). Generalized hypotonia, mild developmental delay.

Investigations: Genetic testing confirmed deletion on chromosome 15q11-q13 (paternal origin).
"""

# Initialize LEAP with custom model and local LLM
obj = leap.LEAP(
    model="all-MiniLM-L12-v2",  # Custom sentence transformer
    llm_client=Gemma3Client(endpoint="http://localhost:11434/api/generate"),
    hp_obo_path="./hp.obo"
)

# Extract phenotypes with reranking
result = obj.convert_ehr(
    content=content,
    rerank=True,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    furthest=True,
    use_weighting=False,
    retrieve_cutoff=0.7,
    rerank_cutoff=0.1
)

print("Extracted HPO terms:", result.final_leap_result)
```

## 📄 License
This project is licensed under the MIT License.

