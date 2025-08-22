# LEAP: LLM-Enhanced Automated Phenotyping
![](https://img.shields.io/badge/python-3.12+-blue.svg)  ![](https://img.shields.io/badge/license-MIT-green.svg)  ![](https://img.shields.io/badge/version-0.1.0-orange.svg)

LEAP (LLM-Enhanced Automated Phenotyping) is a Python package that combines Large Language Models (LLMs) with semantic similarity analysis to automatically extract and map phenotypic descriptions from medical texts to Human Phenotype Ontology (HPO) terms.

## 🌟 Features
+ **Multi-LLM Support**: Compatible with OpenAI GPT-4, Google Gemma3, DeepSeek, Qwen, and Llama models
+ **Automated Phenotype Extraction**: Extract phenotypic descriptions from clinical texts using advanced LLMs
+ **HPO Mapping**: Map extracted phenotypes to standardized HPO (Human Phenotype Ontology) terms using sentence transformers
+ **Gene Ranking**: Prioritize candidate genes based on phenotype profiles using PhenoApt

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
### Prerequisite: configuration file
```bash
# config.yaml
{
    "SAVE_PATH": None,  
    "HP_OBO_PATH": "./hp.obo",  
}
```

`SAVE_PATH`: If set to `None` or an empty string, it will automatically default to `"~/.cache/leap-hpo"`.

`HP_OBO_PATH`: Location of hp.obo

When loading the configuration file, the following priority order is applied:

1. **User-specified path**  
If the `config_path` parameter is provided with a valid path, the function will first attempt to load the configuration file from that location.
2. **Default files in the working directory**  
If `config_path` is not specified, the function will sequentially check the current working directory (`os.getcwd()`) for:
    - `config.yaml`
    - `config.json`
3. **Error if no configuration is found**  
If neither of the above steps succeeds in finding and loading a configuration file, an error will be raised, requiring the user to provide one.

### Basic Usage
```python
from leap import LEAP
from leap.llm_client import OpenAIClient

# Initialize LLM client
llm_client = OpenAIClient(api_key="your-openai-api-key")

# Initialize LEAP
leap = LEAP(
    model="all-MiniLM-L6-v2",  # Sentence transformer model
    llm_client=llm_client,
    cut_off=0.72  # Similarity threshold
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
    furthest=True,      # Retain most specific HPO terms
    use_weighting=False # Use simple top-1 mapping
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

# Use with LEAP
leap_gemma = LEAP(model="all-MiniLM-L6-v2", llm_client=gemma_client)
```

## 📊 Advanced Features
### Weighted Phenotype Mapping
Enable advanced weighting to improve HPO term selection:

```python
result = leap.convert_ehr(
    content=medical_text,
    furthest=True,
    use_weighting=True  # Use top-10 branch analysis for better mapping
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

## 📄 License
This project is licensed under the MIT License.

