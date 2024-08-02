# BioModelComparison

This project compares the effectiveness of deep learning architectures and training protocols on various biological datasets.

## Requirements

This project requires Python 3.9 or later. You can download Python from the [official website](https://www.python.org/downloads/). We recommend using [conda](https://docs.conda.io/en/latest/) to manage your Python environment. The expected size of the project is around 4.1 GB for the datasets and 5.6 GB total.

## Setting up the environment

Download this project to your machine, or clone it using [git](https://git-scm.com/).

Change (`cd`) into the project directory and create a new virtual environment. Using [conda](https://docs.conda.io/en/latest/) on macOS/Linux, you can create a new environment with the required packages by running the following commands in your terminal:

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Downloading data

To access the datasets used in this work, you must set up a [Kaggle](https://www.kaggle.com/) account and download a Kaggle API key. You can do this by following the instructions [here](https://www.kaggle.com/docs/api). Briefly, you must download the API key from your Kaggle account settings and place it in the `~/.kaggle/` directory.

Next you will need to install a rar extraction tool. On Linux, you can do this by running the following command in your terminal:

Linux (e.g., Ubuntu/Debian)

```bash
sudo apt-get install unrar
```

For macOS, you can download `rar` or any other unrar-ing tools. Here is an example using the [Homebrew](https://brew.sh/) package manager:

```bash
brew install rar
```

Once you have your API key and rar, you can download the datasets by running the following commands in your terminal:

```bash
python datasets/download_data.py
```

This will take a few minutes to download the datasets and extract them into the `data/` directory.

## Tutorials

In this repository, we include a few tutorials that demonstrate how to train deep learning models on biological datasets. These tutorials can be found in the `tutorials/` directory.
