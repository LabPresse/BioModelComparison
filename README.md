# BioModelComparison

This project compares the effectivenss of deep learning arcitectures and training protocols on various biological datasets.

## Setting up the environment

Download this project to your machine, or clone it using [git](https://git-scm.com/).

Change (`cd`) into the project directory and create a new virtual environment. Using [conda](https://docs.conda.io/en/latest/) on macOS/Linux, you can create a new environment with the required packages by running the following commands in your terminal:

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Downloading data

To access the datasets used in this work, you must set up a Kaggle account and download a Kaggle API key. You can do this by following the instructions [here](https://www.kaggle.com/docs/api). Breifly, you must download the API key from your Kaggle account settings and place it in the `~/.kaggle/` directory.

Once you have your API key, you can download the datasets by running the following commands in your terminal:

```bash
python download_data.py
```

This will take a few minutes to download the datasets and extract them into the `data/` directory.
