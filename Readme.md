# MishnahBot: A Cross-Lingual RAG Application
![DALL·E 2024-05-22 12 30 55 - A robot studying Mishnah in a yeshiva  The robot, designed with a sleek and modern look, is seated at a traditional wooden desk, surrounded by stacks ](https://github.com/shlomota/MishnahBot/assets/73965390/f627cb44-3836-480c-83fa-ec8beb633a86)


This project demonstrates the creation of a Retrieval-Augmented Generation (RAG) application that interacts with the Mishnah, an ancient Rabbinic text, using AWS Bedrock, LangChain, and ChromaDB. The application supports both English and Hebrew interactions.

## Recent Updates

- **LLM Reranking (2026):** Retrieves top 20 candidate passages from ChromaDB and uses the LLM in a single call to both answer the question and identify which sources were actually used. Cited sources are displayed separately from related-but-not-cited sources.
- **Related Sources Section (2026):** After each answer, a collapsible "Related Sources" section shows topically relevant passages that weren't directly cited in the answer.
- **Upgraded to Claude Sonnet 4.5 (2026):** Migrated from the legacy Claude 3 Sonnet model to `claude-sonnet-4-5` via AWS Bedrock cross-region inference profiles.
- **Migrated to `langchain-aws` + LCEL (2026):** Replaced deprecated `langchain_community.BedrockChat` and `LLMChain` with `langchain_aws.ChatBedrock` and LangChain Expression Language (LCEL), removing dependency on `langchain` and `langchain-classic`.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset](#dataset)
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
    git clone https://github.com/shlomota/MishnahBot.git
    cd MishnahBot
    ```

2. Install the necessary Python packages:
    ```bash
    pip install -r src/requirements.txt
    ```

3. You will need to set up access to AWS Bedrock and enable the Claude Sonnet 4.5 model (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`) via a cross-region inference profile. Alternatively, you can alter the code and use your preferred LLM.

## Dataset

The dataset for this project is the Mishnah, obtained from the Sefaria-Export repository. The dataset includes both English translations and the original Hebrew text.

### Downloading the Dataset
If you want to build the vector db from scratch, run the following commands to download the dataset:
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
tree Mishnah/ | less
```

## Running the RAG Application
To continue building the application you can follow the [notebook](https://github.com/shlomota/MishnahBot/blob/main/Mishnah%20RAG.ipynb).
Alternatively, you can run the streamlit app directly.
```
cd src/
streamlit run app.py
```

## Multilingual RAG Approach
This application supports both Hebrew and English interactions. It uses the following approach:

### Hebrew Mode:
* Input query in Hebrew.
* Translate the query to English.
* Embed the query and retrieve the top 20 relevant passages.
* LLM answers in Hebrew and identifies which passages it used.
* Cited sources and related sources are displayed separately.

### English Mode:
* Input query in English.
* Embed the query and retrieve the top 20 relevant passages.
* LLM answers in English and identifies which passages it used.
* Cited sources and related sources are displayed separately.

## Conclusion
This project highlights the potential of RAG applications in making ancient texts accessible and interactive. By combining modern AI technologies with traditional texts, we can create powerful tools for education and research.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
