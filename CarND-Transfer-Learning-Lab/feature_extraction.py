import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.models import Sequential


#flags = tf.app.flags
#FLAGS = flags.FLAGS

# command line flags
#flags.DEFINE_string('training_file', './vgg-100/vgg_cifar10_100_bottleneck_features_train.p', "Bottleneck features training file (.p)")
#flags.DEFINE_string('validation_file', './vgg-100/vgg_cifar10_100_bottleneck_features_validation.p', "Bottleneck features validation file (.p)")

#flags.DEFINE_integer('epochs', 50, "The number of epochs.")
#flags.DEFINE_integer('batch_size', 256, "The batch size.")




#training_file='./vgg-100/vgg_cifar10_100_bottleneck_features_train.p'
#validation_file='./vgg-100/vgg_cifar10_bottleneck_features_validation.p'

#training_file='./inception-100/inception_cifar10_100_bottleneck_features_train.p'
#validation_file='./inception-100/inception_cifar10_bottleneck_features_validation.p'

training_file='./resnet-100/resnet_cifar10_100_bottleneck_features_train.p'
validation_file='./resnet-100/resnet_cifar10_bottleneck_features_validation.p'




training_file='./vgg-100/vgg_traffic_100_bottleneck_features_train.p'
validation_file='./vgg-100/vgg_traffic_bottleneck_features_validation.p'

#training_file='./inception-100/inception_traffic_100_bottleneck_features_train.p'
#validation_file='./inception-100/inception_traffic_bottleneck_features_validation.p'

#training_file='./resnet-100/resnet_traffic_100_bottleneck_features_train.p'
#validation_file='./resnet-100/resnet_traffic_bottleneck_features_validation.p'

EPOCHS=50
BATCH_SIZE=256

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    print("helo")
    print(training_file)
    X_train, y_train, X_val, y_val = load_bottleneck_data(training_file, validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    nb_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]    
    print(nb_classes, input_shape)

    #inp = Input(shape=input_shape)
    #x = Flatten()(inp)
    #x = Dense(nb_classes, activation='softmax')(x)
    #model = Model(inp, x)
    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    model.fit(X_train, y_train, nb_epoch=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
