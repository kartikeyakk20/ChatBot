import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import json
import nltk
import string
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import tflearn
from textblob import TextBlob
st.set_page_config(page_title = "ChatBot!!!")
st.title("Hi, I am Cheerio. Great to have you here...")
st.header("How can I help you?")
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm Cheerio, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
if 'newWords' not in st.session_state:
    st.session_state.newWords=[]
if 'ourClasses' not in st.session_state:
   st.session_state.ourClasses=[]
if 'data' not in st.session_state:
    f=open('Intent.json')
    st.session_state.data=json.load(f)

lm = WordNetLemmatizer()

response_container = st.container()
colored_header(label='', description='', color_name='blue-30')
input_container = st.container()

if 'ourNewModel' not in  st.session_state :
    documentX = []
    documentY = []

    for intent in st.session_state.data["intents"]:
        for pattern in intent["text"]:
            textnew=TextBlob(pattern)
            pattern=textnew.correct()
            pattern=str(pattern)
            ournewTkns = nltk.word_tokenize(pattern)
            st.session_state.newWords.extend(ournewTkns)
            documentX.append(pattern)
            documentY.append(intent["intent"])


        if intent["intent"] not in st.session_state.ourClasses:
            st.session_state.ourClasses.append(intent["intent"])

    st.session_state.newWords = [lm.lemmatize(word.lower()) for word in st.session_state.newWords if word not in string.punctuation] 
    st.session_state.newWords = sorted(set(st.session_state.newWords))
    st.session_state.ourClasses = sorted(set(st.session_state.ourClasses))

    trainingData = [] 
    outEmpty = [0] * len(st.session_state.ourClasses)
    for idx, doc in enumerate(documentX):
        bagOfwords = []
        text = lm.lemmatize(doc.lower())
        for word in st.session_state.newWords:
            bagOfwords.append(1) if word in text else bagOfwords.append(0)

        outputRow = list(outEmpty)
        outputRow[st.session_state.ourClasses.index(documentY[idx])] = 1
        trainingData.append([bagOfwords, outputRow])
    trainingData = np.array(trainingData,dtype=object)

    x = np.array(list(trainingData[:, 0]))
    y = np.array(list(trainingData[:, 1]))
    print(x)
    print(y)
    net=tflearn.input_data(shape=[None,len(x[0])])
    net=tflearn.fully_connected(net,8)
    net=tflearn.fully_connected(net,8)
    net=tflearn.fully_connected(net,len(y[0]),activation="softmax")
    net=tflearn.regression(net)

    st.session_state.ourNewModel=tflearn.DNN(net)
    st.session_state.ourNewModel.fit(x,y,n_epoch=1000,batch_size=8,show_metric=True)
    st.session_state.ourNewModel.save("chatbot_model")



def ourText(text):
  textnew=TextBlob(text)
  text=textnew.correct()
  text=str(text)
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word) for word in newtkns]
  return newtkns

def wordBag(text, vocab):
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns:
    for idx, word in enumerate(vocab):
      if word == w:
        bagOwords[idx] = 1
  return np.array(bagOwords)

def Pclass(text, vocab, labels):
  bagOwords = wordBag(text, vocab)
  ourResult = st.session_state.ourNewModel.predict(np.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson):
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents:
    if i["intent"] == tag:
      ourResult = random.choice(i["responses"])
      break
  return ourResult




def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        intents = Pclass(user_input, st.session_state.newWords, st.session_state.ourClasses)
        response = getRes(intents, st.session_state.data)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True)
            message(st.session_state["generated"][i])