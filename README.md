# Probing Classifier Scripts (Lynne Swingler Skripsie Project)

This repository contains probing and analysis scripts developed as part of my Skripsie project (`zerosylprobes`). These scripts analyse features from a WavLM model and run different probing architectures to test which one works best. Additionally, the scripts also run L1 verification on the chosen logistic regression probe and PCA is also applied to test different hypotheses and analyse what the hidden features are doing. Finally, a range of visualisation scripts have been included - these were used for the report.

## Project Structure

* `*.py` (e.g., `extractfeatures.py`, `PCA.py`, `train_linear_probe.py`, `utils.py`) — Custom Python scripts for feature extraction, training linear/soft/Gaussian non-linear probes, and analysis.
* `__pycache__/` — Compiled Python bytecode.

## How to Run This Code

Because these scripts rely on the evaluation methodology and data structures from the upstream **ZeroSyl** code, you need to place this folder inside a set up ZeroSyl environment.
To access this repo please use the following link: https://github.com/nicolvisser/ZeroSyl/

## Data Used in This Code
The speech data used in this project comes from Librispeech's open language and speech resources.
dev-clean: for evaluating the probes
train-clean-100: for training the probes

### Step 1: Clone and Set Up ZeroSyl
Follow the instructions in the main repository to set up the environment and download dependencies:
1. Clone the base repository (or use your existing ZeroSyl workspace).
2. Ensure your Python virtual environment (`venv`) is activated.

### Step 2: Add These Probing Scripts
Drop the `zerosylprobes` folder directly into the root directory of your `ZeroSyl` project folder so your file structure looks like this:
```text
ZeroSyl/
│
├── data/
├── zerosyl/
├── zerosylprobes/       <-- Place this folder here
│   ├── extractfeatures.py
│   ├── train_linear_probe.py
│   ├── utils.py
│   └── ...
├── README.md
└── ...
```

### Step 3: Running the code
To run any of the probes, analysis and visualisations from the terminal:
Navigate into your ZeroSyl root directory and then run any of the probing scripts using Python, for example:
```python -m zerosylprobes.extractfeatures``` or ```python zerosylprobes/train_linear_probe.py```
