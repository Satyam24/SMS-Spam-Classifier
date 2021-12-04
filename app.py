import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model_bnb.pkl', 'rb'))

def transform_text(text):
    text_lower = text.lower()
    text_lower_token = nltk.word_tokenize(text_lower)
    text_final = []

    for i in text_lower_token:
        if i.isalnum():
            text_final.append(i)

    text_all = text_final[:]
    text_final.clear()

    for i in text_all:
        if i not in stopwords.words('english') and i not in string.punctuation:
            text_final.append(i)

    text_new = text_final[:]
    text_final.clear()

    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [lemmatizer.lemmatize(word) for word in text_new]

    return " ".join(lemmatized_words)

st.title('SMS Spam Classifier')

input_sms = st.text_area('Enter a message: ')

if st.button('Predict'):


    # Steps
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize

    vector_input = tfidf.transform([transformed_sms])
    # 3. predict

    result = model.predict(vector_input)[0]

    # 4. display result

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
