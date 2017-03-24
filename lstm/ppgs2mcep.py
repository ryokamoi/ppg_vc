import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

FRAME_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 40

PPG_SIZE = 36
HIDDEN_SIZE = 64
MCEP_SIZE = 40

if __name__ == '__main__':
    x_train = np.load('data/trainmfc.npy')
    y_train = np.load('data/targetppg.npy')
    x_test = x_train[:10]
    y_test = y_train[:10]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), \
                      input_shape=(FRAME_SIZE, MFCC_SIZE)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(PHONEME_SIZE))
    model.add(Activation('softmax'))
    
    es = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    mc = ModelCheckpoint('lstm/trainedmodel/weights{epoch:02d}.hdf5', \
                                          monitor='val_loss', save_best_only=False, \
                                          save_weights_only=False, mode='auto')
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, \
                   validation_data=[x_test, y_test], callbacks=[es, mc])
