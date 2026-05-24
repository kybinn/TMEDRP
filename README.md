# TMEDRP

We propose a deep learning framework called TMEDRP (Decoding Tumor-Intrinsic and Microenvironmental Signatures for Clinical Drug Response Prediction), which is designed to predict clinical drug responses by integrating tumor-intrinsic and microenvironmental signatures. 

The framework consists of a pre-training module to learn robust representations from multi-omics data, followed by a fine-tuning module tailored for accurate downstream drug response prediction.

## Quick Start

TMEDRP is implemented in Python and based on the open-source deep learning library **PyTorch 2.4.1**. To set up the environment, we recommend using `conda` or `pip`.

### 1. Environment Configuration

**Using Conda:**
```bash
# Create an environment from the provided yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate environment
```

**Using Pip:**
If you prefer using pip, you can install the required dependencies directly via:
```bash
pip install -r requirements.txt
```

### 2. Running the Model

The pipeline is split into three main stages: pre-training, fine-tuning, and evaluation. 

#### 🔹 Phase 1: pre-training / Retraining
To start the model pre-training phase, open `main.py` and set `retrain_flag = True`. Then execute the script:
```bash
python main.py
```

#### 🔹 Phase 2: Fine-Tuning / Running
To switch to the fine-tuning and prediction phase, open `main.py` and set `retrain_flag = False`. Then execute the script:
```bash
python main.py
```



## Repository Structure

```text
├── utils/                  # Utility functions for data processing and evaluation
├── dataload.py             # Data loading and preprocessing pipeline
├── environment.yaml        # Conda environment configuration file
├── requirements.txt        # Pip dependencies list
├── train_params.json       # Hyper-parameters configuration file
├── models.py               # Model architecture definitions
├── pretraining.py          # Script for model pre-training
├── finetuning.py           # Script for downstream fine-tuning
└── main.py                 # Main entry script (Controls training/fine-tuning via retrain_flag)
```

## Data

Please ensure your gene expression data and drug response datasets are formatted properly before running the scripts. 
The genomics datasets used in this study are available at: https://github.com/liuxuan666/TransDRP.
The TME-related data will be uploaded promptly after the peer review process is completed.

