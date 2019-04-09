import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    if (window_size>=len(series)):
        print('Error, windows_size must be smaller than the length of the series')
        return 0

    # containers for input/output pairs
    X = []
    y = []

    #Stores every input sequence and the respective output value in arrays
    for e in range(window_size,len(series)):
        X += [series[e-window_size:e]]
        y += [series[e]]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    modelo = Sequential()
    #Add LSTM layer with 5 hidden nodes
    modelo.add(LSTM(5, input_shape=(window_size,1)))
    #Fully connected layer with one node
    modelo.add(Dense(1))

    return modelo


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    textSet = list(set(text))
    for i in textSet:
        #Ignore characters in the punctuation array or if they are part of the lower case alphabet (ASCii from 97 to 122)  
        if  not((i in punctuation) or ((ord(i)>=97) and (ord(i)<=122))):
            text = text.replace(i, ' ')  
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):

    if (window_size>=len(text)):
        print('Error, windows_size must be smaller than the length of the text')
        return 0

    # containers for input/output pairs
    inputs = []
    outputs = []

    #Define number of steps required in function of window_size and step size
    steps = (len(text)-window_size)/(step_size)+1
    #Stores every input sequence and the respective output value in arrays
    for i in range(int(np.floor(steps))):
        inputs += [text[(i*step_size):(window_size+i*step_size)]]
        outputs += [text[window_size+i*step_size]]
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    modelo = Sequential()
    modelo.add(LSTM(200, input_shape=(window_size, num_chars)))
    modelo.add(Dense(num_chars, activation='linear'))
    modelo.add(Activation('softmax'))
    return modelo
