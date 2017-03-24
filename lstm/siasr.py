import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

FRAME_SIZE = 200
BATCH_SIZE = 2
EPOCHS = 40
STATEFUL = True

MFCC_SIZE = 40
HIDDEN_SIZE = 64
PHONEME_SIZE = 36

class ResetStates(Callback):
    def __init__(self):
        self.batches = 0

    def on_batch_end(self, batch, logs={}):
        if self.batches % BATCH_SIZE == 0:
            self.model.reset_states()
        self.batches += 1

if __name__ == '__main__':
    x_train = np.load('data/trainmfc.npy')
    y_train = np.load('data/trainppg.npy')
    x_test = x_train[:10]
    y_test = y_train[:10]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, \
                      stateful=STATEFUL), \
                      batch_input_shape=(BATCH_SIZE, FRAME_SIZE, MFCC_SIZE)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, \
                      stateful=STATEFUL), \
                      batch_input_shape=(BATCH_SIZE, FRAME_SIZE, HIDDEN_SIZE)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, \
                      stateful=STATEFUL), \
                      batch_input_shape=(BATCH_SIZE, FRAME_SIZE, HIDDEN_SIZE)))
    model.add(Dropout(0.5))
    model.add(Dense(PHONEME_SIZE))
    model.add(Activation('softmax'))
    
    rs = ResetStates()
    es = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    mc = ModelCheckpoint('lstm/trainedmodel/weights{epoch:02d}.hdf5', \
                                          monitor='val_loss', save_best_only=False, \
                                          save_weights_only=False, mode='auto')
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, \
                   validation_data=[x_test, y_test], callbacks=[rs, es, mc], shuffle=False)
