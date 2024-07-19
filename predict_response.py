# Bibliotecas de pré-processamento de dados de texto
import nltk

import json
import pickle
import numpy as np
import random


import tensorflow
from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m"]
#Biblioteca load_model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

#Ola meu nome é Pedro e eu estou entendiado hoje

# cria as 3 classes responsaveis por execultar a o treinamenot
def preprocess_user_input(user_input):
    #recebe a entrada do usuario, tokeniza
    input_word_token_1 = nltk.word_tokenize(user_input)
    #converte o texto tokenizado em palavras tronco
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    #cria uma lista ordenada 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Codificação dos dados de entrada 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

def bot_class_prediction(user_input):
    #chamamos a função acima com o texto do usuario
    inp = preprocess_user_input(user_input)
    #preve a etiqueta e armazena
    #array de classes previstas e suas possibilidades
    prediction = model.predict(inp)
    #argmax() encontra o valor maximo, retornando a previsao da
    #etiqueta ou classe
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

#resposta para o usuario
def bot_response(user_input):
     #resultado da função acima na variavel abaixo ex "saudação"
   predicted_class_label =  bot_class_prediction(user_input)
   predicted_class = classes[predicted_class_label]

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        bot_response = random.choice(intent['responses'])
        return bot_response

#2 criar o processo de entrada da pergunta do usuario
print("Oi, eu sou a Estela , como posso ajudar?")

while True:
    user_input = input("Digite sua mensagem aqui:")
    print("Entrada do Usuário: ", user_input)
    #4 cria a resposta do robo
    response = bot_response(user_input)
    print("Resposta do Robô: ", response)

