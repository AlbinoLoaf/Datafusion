# DataFusion

Ensure `util.py` and a `data/` folder (containing datasets) are in the same directory as `notebook.ipynb` and `utils.py`.

```shell
├─  data/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── resnet50_finetuned.pth
└── util.py
```

## Setup & Execution

```shell
# 1. Environment Setup
conda create --name df_4 python=3.11 -y
conda activate df_4

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch (Run cells top-to-bottom)
jupyter notebook
```

Running it in pycharm, and vscode is also possible.
