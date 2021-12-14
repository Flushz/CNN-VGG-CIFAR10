# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-12-14T20:50:30.385404Z","iopub.execute_input":"2021-12-14T20:50:30.385732Z","iopub.status.idle":"2021-12-14T20:50:30.393233Z","shell.execute_reply.started":"2021-12-14T20:50:30.385698Z","shell.execute_reply":"2021-12-14T20:50:30.392261Z"},"jupyter":{"outputs_hidden":false}}
import sys
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

# %% [code] {"execution":{"iopub.status.busy":"2021-12-14T20:50:30.395487Z","iopub.execute_input":"2021-12-14T20:50:30.395990Z","iopub.status.idle":"2021-12-14T20:50:30.414795Z","shell.execute_reply.started":"2021-12-14T20:50:30.395944Z","shell.execute_reply":"2021-12-14T20:50:30.413906Z"},"jupyter":{"outputs_hidden":false}}


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# %% [code] {"execution":{"iopub.status.busy":"2021-12-14T20:50:30.416105Z","iopub.execute_input":"2021-12-14T20:50:30.416425Z","iopub.status.idle":"2021-12-14T20:50:30.426705Z","shell.execute_reply.started":"2021-12-14T20:50:30.416386Z","shell.execute_reply":"2021-12-14T20:50:30.425861Z"},"jupyter":{"outputs_hidden":false}}
# scale pixels


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

# %% [code] {"execution":{"iopub.status.busy":"2021-12-14T20:50:30.429259Z","iopub.execute_input":"2021-12-14T20:50:30.429600Z","iopub.status.idle":"2021-12-14T20:50:30.439389Z","shell.execute_reply.started":"2021-12-14T20:50:30.429560Z","shell.execute_reply":"2021-12-14T20:50:30.438378Z"},"jupyter":{"outputs_hidden":false}}


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-14T20:50:30.441112Z","iopub.execute_input":"2021-12-14T20:50:30.441445Z","iopub.status.idle":"2021-12-14T20:50:30.457463Z","shell.execute_reply.started":"2021-12-14T20:50:30.441389Z","shell.execute_reply":"2021-12-14T20:50:30.456310Z"},"jupyter":{"outputs_hidden":false}}


def define_model():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    with tpu_strategy.scope():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))

        model.add(Dense(10, activation='softmax'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(
            optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# %% [code] {"execution":{"iopub.status.busy":"2021-12-14T20:50:30.458726Z","iopub.execute_input":"2021-12-14T20:50:30.458984Z","iopub.status.idle":"2021-12-14T21:11:25.298538Z","shell.execute_reply.started":"2021-12-14T20:50:30.458958Z","shell.execute_reply":"2021-12-14T21:11:25.297700Z"},"jupyter":{"outputs_hidden":false}}
if __name__ == '__main__':
    # load dataset
    trainX, trainY, testX, testY = load_dataset()

    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)

    # define model
    model = define_model()

    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64,
                        validation_data=(testX, testY), verbose=1)

    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)

    model.save('./final_model.h5')
