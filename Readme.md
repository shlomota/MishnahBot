# MishnahBot: A Cross-Lingual RAG Application

This project demonstrates the creation of a Retrieval-Augmented Generation (RAG) application that interacts with the Mishnah, an ancient Rabbinic text, using AWS SageMaker, AWS Bedrock, LangChain, and ChromaDB. The application supports both English and Hebrew interactions.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset](#dataset)
- [Vectorization and Storage](#vectorization-and-storage)
- [Running the RAG Application](#running-the-rag-application)
- [Multilingual RAG Approach](#multilingual-rag-approach)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to make ancient Jewish texts more accessible by enabling semantic search and finding related sources. The same approach can be applied to any other collection of texts. The project uses state-of-the-art AI technologies to achieve efficient and accurate retrieval and generation of text.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/shlomota/Multilingual-RAG-Demo.git
    cd Multilingual-RAG-Demo
    ```

2. Set up your AWS credentials:
    ```bash
    aws configure
    ```

3. Install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset for this project is the Mishnah, obtained from the Sefaria-Export repository. The dataset includes both English translations and the original Hebrew text.

### Downloading the Dataset

Run the following commands to download the dataset:
```bash
git init sefaria-json
cd sefaria-json
git sparse-checkout init --cone
git sparse-checkout set json
git remote add origin https://github.com/Sefaria/Sefaria-Export.git
git pull origin master

mkdir -p new_directory
find Mishnah/Seder* -name "merged.json" -exec cp --parents \{\} new_directory/ \;
sudo apt install tree
tree Mishna/ | less
