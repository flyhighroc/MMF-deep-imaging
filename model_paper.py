import numpy as np
np.random.seed(42)

from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Dropout, Reshape, BatchNormalization
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

import io_data
from helpers import remove
from helpers import normalize


class Optic_Transformer(object):
    """docstring for Optic_Transformer"""

    def __init__(self,name):
        self.inp_sz = 96
        self.op_sz = 36

        self.threshold = 127

        self.batch_size = 100
        self.train_samples = 32775
        self.test_samples = 8194
        self.steps_per_epoch = self.train_samples // self.batch_size + 1
        self.validation_steps = self.test_samples // self.batch_size + 1

        self.name = ''.join(['mnist', '_',name])
        self.best_model_path = ''.join(
            ['models/', self.name, '_best', '.hdf5'])
        self.last_model_path = ''.join(
            ['models/', self.name, '_last', '.hdf5'])
        self.log_path = ''.join(['logs/', self.name, '_logs', '.csv'])
        self.graph_path = ''.join(['logs/', self.name, '_graph', '.png'])

    def get_data(self,dataset='mnist'):
        if dataset=='mnist':
            (X_train, y_train), (X_test, y_test) = io_data.load_data()m
            return X_train, X_test, y_train, y_test
        else:
            X,y = io_data.read_random_pattern()
            return X,y

    def get_model(self, dropout=False):

        input_img = Input(shape=(self.inp_sz, self.inp_sz, 1))
        x = input_img

        filters = [5, 10, 24]
        strides = [1, 2, 1]
        for f, s in zip(filters, strides):
            x = Conv2D(f, kernel_size=(3, 3), strides=(
                s, s), activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)

        x = Flatten()(x)
        x = Activation(activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(self.op_sz * self.op_sz, activation='sigmoid')(x)

        prediction = Reshape(target_shape=(self.op_sz, self.op_sz, 1))(x)
        model = Model(inputs=[input_img], outputs=[prediction])
        model.summary()
        return model

    def build_model(self, lr=1e-2):

        model = self.get_model()
        opt = RMSprop(lr)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_accuracy']
                      )
        return model

    def get_callbacks(self):
        checkpoint = ModelCheckpoint(self.best_model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto', baseline=None)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                      verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=1e-7)
        csv_logger = CSVLogger(self.log_path, append=True)
        return [checkpoint, csv_logger,
                reduce_lr, early_stopping
                ]

    def train(self, lr=1e-2, epochs=1):

        remove(self.log_path)
        model = self.build_model(lr)
        model.summary()

        X_train, X_test, y_train, y_test = self.get_data()
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=self.get_callbacks(),
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  initial_epoch=0,
                  steps_per_epoch=None, validation_steps=None)

        save_model(model, self.last_model_path)

    def continue_train(self, lr=1e-4, epochs=1, resume_model='last'):

        if resume_model == 'last':
            model = load_model(self.last_model_path)
        else:
            model = load_model(self.best_model_path)
        model.summary()

        X_train, X_test, y_train, y_test = self.get_data()
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=self.get_callbacks(),
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  # initial_epoch=start_epoch,
                  steps_per_epoch=None, validation_steps=None)
        save_model(model, self.last_model_path)

    def evaluate(self, visualize=False, dataset='mnist'):

        if dataset == 'mnist':
            print('mnist')
            X_train, X_test, y_train, y_test = self.get_data()
        else:
            print('random')
            X_test,y_test = self.get_data(dataset)

        model = load_model(self.best_model_path)
        y_pred_prob = model.predict(x=X_test,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    steps=None
                                    )

        if visualize:
            y_pred = y_pred_prob.copy()
            y_pred = normalize(y_pred, 0, 255)
            y_pred = y_pred > self.threshold
            y_pred = y_pred * 255.0

            y_true = y_test.copy()
            y_true = y_true * 255.0

            y_pred = y_pred.reshape(-1, self.op_sz, self.op_sz)
            y_true = y_true.reshape(-1, self.op_sz, self.op_sz)

            from helpers import write_imgs_serially
            side_by_side = np.hstack([y_true, y_pred])
            write_imgs_serially(
                side_by_side, base_path='tmp/' + self.name + '_op/'+dataset+'/')

        from sklearn.metrics import accuracy_score, mean_absolute_error
        from sklearn.metrics import mean_squared_error

        y_true = y_test.copy().reshape(-1, 1)
        y_pred = y_pred_prob.copy().reshape(-1, 1)
        y_pred = y_pred > 0.5
        y_pred = y_pred * 1

        acc = accuracy_score(y_true, y_pred)
        print('ACC : ', acc)
        mae = mean_absolute_error(y_true, y_pred)
        print('MAE : ', mae)
        mse = mean_squared_error(y_true, y_pred)
        print('MSE : ', mse)

        from helpers import plot_model_history
        addn_info = 'Acc : ' + str(round(acc, 3))
        plot_model_history(self.log_path, self.graph_path, addn_info=addn_info)

if __name__ == '__main__':

    clf = Optic_Transformer(name='m0')
    # clf.train(lr=1e-3, epochs=100)
    clf.evaluate(visualize=True,dataset='mnist')
    clf.evaluate(visualize=True,dataset='rp')
