# Bibliotecas de treinamento do modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
#importa a classe data_perpocessing
from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    #camada densa com 128 filtros  qtd de frase          metodo de filtragem
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    #reaproveitar o que foi disperdiçado da camada anterior em 50%
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #cria a ultima camada com a quantidade de filtros da mesma quantidade de tag/classes
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile o modelo com a função de perda e otimização para reduzir as perdas
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    # Ajuste e salve o modelo
                        #dados do treinamento
                                        #treina o modelo em 200 vezes 
                                                    #tamanho do lote
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    #1 explica , salva o modelo e chama as funções
    model.save('chatbot_model.h5', history)
    print("Modelo Criado e Salvo")


# Chamando os métodos para treinar o modelo
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

