# FAQ Chatbot using NLP (Python + NLTK + TF-IDF + Gradio)

## Project Overview

This project builds a **FAQ Chatbot** that automatically answers user
questions by matching them with the most similar question in a dataset.

The chatbot uses: - **NLTK** for text preprocessing - **TF-IDF
Vectorization** for converting text to numerical form - **Cosine
Similarity** to find the most similar FAQ - **Gradio** to create a
simple chatbot web interface

## Features

-   Preprocesses user text (lowercase, tokenization, stopword removal)
-   Uses **machine learning similarity matching**
-   Returns the most relevant FAQ answer
-   Includes both **terminal chat mode and web UI**
-   Uses a **CSV dataset for FAQs**

## Project Structure

    project/
    │
    ├── chatbot.py
    ├── faq_dataset.csv
    └── README.md

## Installation

Install required libraries:

``` bash
pip install nltk scikit-learn gradio pandas
```

## Dataset (CSV File)

The chatbot uses **faq_dataset.csv** which contains:

| question \| answer \|

\|--------\|--------\| What is your return policy? \| You can return any
product within 30 days of purchase. \|

Example CSV:

    question,answer
    What is your return policy?,You can return any product within 30 days of purchase.
    How long does shipping take?,Shipping usually takes 3 to 5 business days.

## Connecting CSV to the Code

Replace the dictionary with CSV loading:

``` python
import pandas as pd

df = pd.read_csv("faq_dataset.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()
```

## Running the Chatbot

Run the Python file:

``` bash
python chatbot.py
```

Then:

-   Terminal chatbot will start
-   Gradio web interface will open

## Example Chat

User:

    How do I track my order?

Bot:

    You can track your order using the tracking link sent to your email.

## Technologies Used

-   Python
-   NLTK
-   Scikit-learn
-   TF-IDF
-   Cosine Similarity
-   Gradio

## Future Improvements

-   Use **BERT embeddings**
-   Add **voice chatbot**
-   Connect with **database**
-   Deploy using **Streamlit or Flask**
