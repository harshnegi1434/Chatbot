# importing relevant libraries
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

from tkinter import *

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# loading the Dataset : intents.json
intents = json.loads(open('C:\Programming\prgms\projectI\intents.json').read())

# loading the words , classes and model file
words = pickle.load(open('C:\Programming\prgms\projectI\words.pkl', 'rb'))
classes = pickle.load(open('C:\Programming\prgms\projectI\classes.pkl', 'rb'))
model = load_model('C:\Programming\prgms\projectI\chatbot_model.h5')


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    # return bag of words array : 0 or 1 for each word in the bag that exists in the sentence
    return sentence_words


def bag_of_words(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    # sourcery skip: inline-immediately-returned-variable, list-comprehension
    # filter out predictions below a threshold
    bow = bag_of_words(sentence)
    rese = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(rese) if r > error_threshold]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    question = predict_class(text)
    return get_response(question, intents)


# creating GUI with tkinter
base = Tk()
base.title("Chatbot - TWILIGHT")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# create chat window
Chatlog = Text(base, bd="0", bg="white", height="8", width="50", font="Arial")
Chatlog.config(state=DISABLED)

# bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=Chatlog.yview, cursor="heart")
Chatlog['yscrollcommand'] = scrollbar.set

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chatlog.config(state=NORMAL)
        Chatlog.insert(END, f"You: {msg}" + '\n\n')
        Chatlog.config(foreground="#442265", font=("Verdana", 12))
        res = chatbot_response(msg)
        Chatlog.insert(END, f"Twilight: {res}" + '\n\n')
        Chatlog.config(state=DISABLED)
        Chatlog.yview(END)

# create button to send message
Sendbutton = Button(base, font=("Verdana", 12, 'bold'), bd="0", bg="#32de97", 
                    text="Send", width="12", height="5",
                    activebackground="#3c9d9b", fg="#ffffff", command=send)

# create the box to enter message
EntryBox = Text(base, bd="0", bg="white", width="29", height="5", font="Arial")

# place all components on the screen
scrollbar.place(x=376, y=6, height=386)
Chatlog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=18, y=401, height=90, width=265)
Sendbutton.place(x=6, y=401, height=90)

base.mainloop()
