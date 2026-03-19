

import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

nltk.download('punkt')
nltk.download('stopwords')

faq_data = {
    "What is your return policy?": "You can return any product within 30 days of purchase.",

    "How long does shipping take?": "Shipping usually takes 3 to 5 business days.",

    "Do you offer international shipping?": "Yes, we ship to many countries worldwide.",

    "How can I track my order?": "You can track your order using the tracking link sent to your email.",

    "What payment methods do you accept?": "We accept credit cards, debit cards, and PayPal.",

    "How can I contact customer support?": "You can contact our support team via email or live chat.",

    "Do you offer refunds?": "Yes, refunds are processed after we receive the returned item."
}

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

import nltk
nltk.download('punkt_tab')

questions = list(faq_data.keys())
answers = list(faq_data.values())

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):

    user_input_processed = preprocess(user_input)

    user_vector = vectorizer.transform([user_input_processed])

    similarity = cosine_similarity(user_vector, X)

    best_match_index = np.argmax(similarity)

    best_score = similarity[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I couldn't understand your question."

    return answers[best_match_index]

while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    response = chatbot_response(user_input)

    print("Bot:", response)

def chat_interface(message, history):

    response = chatbot_response(message)

    return response


chatbot = gr.ChatInterface(
    fn=chat_interface,
    title="FAQ Chatbot",
    description="Ask questions about our product and services."
)

chatbot.launch()