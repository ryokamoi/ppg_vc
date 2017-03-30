import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

FRAME_SIZE = 2000
BATCH_SIZE = 32
EPOCHS = 60
STATEFUL = False

MFCC_SIZE = 40
HIDDEN_SIZE = 128
PHONEME_SIZE = 36

if __name__ == '__main__':
    x_all = np.load('')
    y_all = np.load('')
    testsize = np.shape(x_all)[0]//10
    testsize = testsize//BATCH_SIZE * BATCH_SIZE
    trainsize = (np.shape(x_all)[0]-testsize) // BATCH_SIZE * BATCH_SIZE
    x_train = x_all[-trainsize:]
    y_train = y_all[-trainsize:]
    x_test = x_all[:testsize]
    y_test = y_all[:testsize]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE,  \
                      return_sequences=True), input_shape=(None, MFCC_SIZE)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Dense(PHONEME_SIZE))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    mc = ModelCheckpoint('', \
                                          monitor='val_loss', save_best_only=False, \
                                          save_weights_only=False, mode='auto')
    lh = LossHistory()
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,\
                   validation_data=[x_test, y_test], callbacks=[es, mc], shuffle=True)
