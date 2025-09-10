# BinRAG

Implementation of BinRAG: A RAG-based System for Function Name Inference on Stripped Binaries.

## Overview

BinRAG combines machine learning techniques with assembly language embedding model to analyze binary code. 
This project is based on the ktrans embedding model.

The embedding model can be modified to suit the user's needs. However, additional steps are required to match the input format required by each embedding model.

## Setup

### Prerequisites

- PyTorch
- PyTorch Lightning
- Transformers
- NumPy
- Additional dependencies in requirements.txt (if available)

### Embedding Model Setup

This project uses the ktrans pre-trained model for assembly code embedding. To set up the embedding model:

1. Download the pre-trained ktrans model from [ktrans repository](https://github.com/Learner0x5a/kTrans-release)
2. Place the downloaded model file (`ktrans-model.ckpt`) in the `pretrained/` directory:
   ```
   binRAG/
   ├── models/
   │   └── ktrans-model.ckpt    # Place your downloaded model here
   ├── AsmLM/
   ├── evaluation/
   └── ...
   ```

**Note**: The current codebase is specifically designed for the ktrans model architecture. If you wish to use a different embedding model, you may need to modify the input preprocessing and model loading components accordingly.

## Key Components

### Core Files

- `build_db.py` - Database construction utilities
- `inference_api.py` - Function name inference with OPENAI API
- `inference_llama.py` - Function name inference with Llama using Ollama

### Data Processing
- **deduplicate_functions.py**: Deduplicate functions
- **remove_sub_functions.py**: Remove non-symbol functions
- **remove_zn_functions.py**: Remove mangled functions

## Sample Database
You can download sample database in our google drive.
https://drive.google.com/file/d/1qDyVOjEsRqJnfwCN2SY9Sq3zxORMQ_3z/view?usp=sharing


## Usage

### Build DB
```bash
python build_db.py \
  --model_path ./models/ktrans-model.ckpt \
  --input-dir /path/to/data \
  --output_dir /path/to/save/db
```

### Inference
```bash
python inference_api.py \
  --model_path ./models/ktrans-model.ckpt \
  --db_path /path/to/function/database \
  --c_dir /path/to/c/files \
  --pickle_dir /path/to/pickle/files

python inference_llama.py \
  --model_path ./models/ktrans-model.ckpt \
  --db_path /path/to/function/database \
  --c_dir /path/to/c/files \
  --pickle_dir /path/to/pickle/files \
  --ollama_model "your/model/name" \
  --k 5
```

## Directory Structure

```
binRAG/
├── AsmLM/                   # Core AsmLM module for ktrans
│   ├── dataloader/          # Data loading utilities
│   └── model/               # Model implementations utilities
├── evaluation/              # Evaluation tools and metrics
├── preprocess/              # Data preprocessing scripts
├── db/                      # Database files
└── pretrained/              # pre-trained model
```

## Evaluation

The project includes evaluation tools based on CodeWordNet for assessing model performance on assembly code understanding tasks.
