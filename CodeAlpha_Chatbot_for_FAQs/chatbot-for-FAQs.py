# Import Libraries
import nltk
import string
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Load FAQ dataset from CSV
df = pd.read_csv("faq_dataset.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()


# Stopwords
stop_words = set(stopwords.words('english'))


# Text preprocessing function
def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Preprocess questions
processed_questions = [preprocess(q) for q in questions]


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_questions)


# Chatbot response function
def chatbot_response(user_input):

    user_input_processed = preprocess(user_input)

    user_vector = vectorizer.transform([user_input_processed])

    similarity = cosine_similarity(user_vector, X)

    best_match_index = np.argmax(similarity)

    best_score = similarity[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I couldn't understand your question."

    return answers[best_match_index]


# Terminal Chat Mode
def terminal_chat():

    print("FAQ Chatbot Started (type 'exit' to quit)\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        response = chatbot_response(user_input)

        print("Bot:", response)


# Gradio Web Interface
def chat_interface(message, history):

    response = chatbot_response(message)

    return response


# Launch Gradio Chatbot
chatbot = gr.ChatInterface(
    fn=chat_interface,
    title="FAQ Chatbot",
    description="Ask questions about our product and services."
)


if __name__ == "__main__":

    # Run terminal chat first
    terminal_chat()

    # Then launch web UI
    chatbot.launch()