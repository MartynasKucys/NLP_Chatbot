# NLP_Chatbot

Uses spacy's pre-trained large english model to tokenize text. Passes the vector to a pytorch neural network to predict the intent of the inquiry.

data set used: https://www.kaggle.com/code/atinsaki/chatbot-using-given-responses/data

## Libraries used:

- pytorch
- spacy
- numpy

## Demo

```
Say goodbye to quit.

	You: hi
	ChatBot: Hi human, please tell me your GeniSys user

prob: 1.0 | tag: Greeting


	You: what is your name
	ChatBot: Call me Geni

prob: 0.996103048324585 | tag: NameQuery


	You: you dont know this
prob: 0.6570475101470947
Chatbot: I don't understand.


	You: see ya later
	ChatBot: See you later

prob: 0.9999645948410034 | tag: GoodBye
```
