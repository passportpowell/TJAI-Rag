### Test Question

tell me about the Modelling Results for Hydrogen Grid Flows


# Joule Model CSV RAG System

A local Retrieval-Augmented Generation (RAG) system for querying the Joule Model Report CSV file using semantic search.

## Overview

This tool allows you to search through the Joule Model Report data using natural language queries. It converts your CSV data into vector embeddings and uses semantic search to find the most relevant entries based on your queries.

## Installation

### Requirements

- Python 3.7+
- pandas
- numpy
- faiss-cpu
- sentence-transformers

### Setup

1. Install the required packages:

```bash
pip install pandas numpy faiss-cpu sentence-transformers
```

2. Save the script as `joule_rag.py` in your preferred directory.

3. Update the CSV path using one of these methods:
   - Edit the `csv_path` variable in the script
   - Use the `--csv` command-line argument (recommended)

## Usage

### Quick Start

```bash
# Basic usage with default CSV path
python joule_rag.py

# Specify a different CSV file
python joule_rag.py --csv "path/to/your/file.csv"

# Force rebuild the index (after updating your CSV)
python joule_rag.py --force-rebuild
```

### Step-by-Step Guide

1. **Prepare Your CSV File**:
   - Ensure your CSV has the expected columns (objective_title, task_title, etc.)
   - Place it in an accessible location

2. **Run the Script**:
   ```bash
   python joule_rag.py --csv "D:\OneDrive\Python Scripts\Joule Model Report.csv"
   ```

3. **Initial Setup Process**:
   - The system loads your CSV data
   - Downloads the sentence transformer model (first time only)
   - Builds the vector index (first time only)
   - Launches the interactive query session

4. **Query Interface**:
   ```
   === Joule Model Report RAG System ===
   Type your query or 'exit' to quit

   Enter query: 
   ```

5. **Example Queries**:
   ```
   Enter query: renewable energy projects
   ```
   ```
   Enter query: climate initiatives in France
   ```
   ```
   Enter query: carbon reduction strategies --detailed
   ```

### Query Options

| Command/Option | Description |
|----------------|-------------|
| `your query -d` | Show detailed results including all fields |
| `your query --detailed` | Same as above |
| `exit` or `quit` or `q` | Exit the program |

### Example Results

```
Result #1 (Score: 0.8723)
Objective: Renewable Energy Development
Task: Solar Implementation
Sub-task: Policy framework development
Summary: Framework for implementing solar energy solutions...

Result #2 (Score: 0.7865)
Objective: Clean Energy Transition
Task: Renewable Integration
Sub-task: Funding mechanisms
Summary: Establishing financial support for renewable...
```

With `-d` flag, you'll see all fields including `country`, `final_copy`, etc.

## Index Management

The system creates a persistent index to enable fast startup on subsequent runs.

### How the Index Works

- **First Run**: System builds a full index (may take a few minutes)
- **Later Runs**: System loads the existing index (takes seconds)
- **Index Location**: `rag_index` folder in the same directory as your CSV

### Updating the Index

You can also programmatically force a rebuild:

```python
rag.build_index(force_rebuild=True)
```

Or int he terminal:

```bash
python joule_rag.py --force-rebuild
```



## Advanced Customization

### Changing the Embedding Model

The default model is `all-MiniLM-L6-v2`. To use a different model, modify:

```python
def load_model(self, model_name: str = "all-MiniLM-L6-v2"):
```

Some alternatives:
- `all-mpnet-base-v2` (higher quality, slower)
- `all-MiniLM-L12-v2` (good balance)
- `paraphrase-multilingual-MiniLM-L12-v2` (for non-English data)

### Modifying Search Behavior

1. **Number of Results**:
   ```python
   # In query() method:
   def query(self, query_text: str, n_results: int = 5):
   ```

2. **Fields Used for Search**:
   Edit the `_prepare_text_for_embedding` method to include/exclude fields

3. **Result Formatting**:
   Customize the `format_results` method to change how results are presented

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Error loading CSV"** | Check the file path and ensure proper read permissions |
| **Slow index building** | Normal for first run; will be much faster on subsequent runs |
| **Out of memory errors** | Try a smaller model or reduce batch size in encoding |
| **Poor search results** | Ensure your queries are specific and contain relevant terms |
| **Index not updating** | Use `--force-rebuild` after changing your CSV |

## Performance Tips

- For large CSVs (>10k rows), consider using a faster but slightly less accurate model
- Keep index persistent between runs for faster startup
- For deployment scenarios, consider using IVF index types in FAISS for better scaling
- Regular queries take milliseconds once the index is loaded

## Code Structure

The core functionality is contained in the `LocalRAGSystem` class:

- `__init__`: Initialization with CSV path and index directory
- `load_data()`: CSV loading with pandas
- `load_model()`: Loads the sentence transformer
- `_prepare_text_for_embedding()`: Combines fields for semantic representation
- `build_index()`: Creates/loads FAISS index
- `query()`: Performs semantic search
- `format_results()`: Human-readable output formatting
- `interactive_query()`: CLI interface