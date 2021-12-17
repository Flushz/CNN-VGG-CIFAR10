import sys
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def define_model():
    config = [64, 'MP', 128, 128, 'MP', 256, 256, 256,
              'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']
    dropout_rate = 0.2

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    for cfg in config:
        if cfg == 'MP':
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(dropout_rate))
            if dropout_rate < 0.5:
                dropout_rate += 0.05
        else:
            model.add(Conv2D(64, (3, 3), activation='relu',
                             kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


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


if __name__ == '__main__':
    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = prep_pixels(trainX, testX)

    model = define_model()

    datagen = ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    it_train = datagen.flow(trainX, trainY, batch_size=64)

    steps = int(trainX.shape[0] / 64)
    history = model.fit(it_train, steps_per_epoch=steps,
                        epochs=200, validation_data=(testX, testY), verbose=1)

    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

    summarize_diagnostics(history)

    model.save('./model.h5')
