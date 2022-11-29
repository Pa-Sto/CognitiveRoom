import json
import Datasets
from time import process_time, strftime
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from os.path import isdir
from os import mkdir
import helper_functions

params = {  # Choose Model Parameter which are used for training
    'model_name': 'AnimalRoom',
    'model': Datasets.AnimalCognitiveRoom(),
    'epochs': 1000,
    'Number_Samples': 5000,
    'Alternate': 0.15,
    'batch_size': 10,
    'input_size': 7,
    'output_size': 32,
    'hidden_layers': 1,
    'bottleneck': 1,
    'SR': False,
    't': 10,
    'df': 1.0
}


def main(tensorboard_log=True):
    # Creating Folder for each Model
    timestr = strftime("%Y.%m.%d-%H:%M:%S")
    folder_name = timestr + '(' + params['model_name'] + '_Epochs:' + str(params['epochs']) + ')'

    modelsfolder = 'Models'

    # Create Training_Data
    if not params['SR']:
        if params['model_name'] is 'AnimalRoom':
            x_train, y_train = helper_functions.get_sequences_AnimalDataSet(params['Number_Samples'],
                                                                            alternate=params['Alternate'])
    else:
        if params['model_name'] is 'AnimalRoom':
            x_train, y_train = helper_functions.get_sequences_fromSR_AnimalDataSet(params['Number_Samples'],
                                                                                   params['t'],
                                                                                   params['df'],
                                                                                   alternate=params['Alternate'])
    # build model
    model = Sequential()
    model.add(Dense(params['input_size'], activation=tf.nn.relu))  # input layer
    if params['hidden_layers'] < 1:        hidden_layers = 1  # minimum 1 hidden layer
    for h_layer in range(1, params['hidden_layers'] + 1):
        # fully connected layer
        model.add(Dense(int(params['input_size'] * params['bottleneck']), activation=tf.nn.relu))
    model.add(Dense(params['output_size'], activation=tf.nn.softmax))  # output layer

    print('Compiling model')
    # pass parameters
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # config tensorboard
    tb = None
    if tensorboard_log:
        # TODO tensorboard mehr metrics uebergeben
        tb = TensorBoard(log_dir="./Tensorboard/logs/" + folder_name)

    print('Starting training')
    history = model.fit(x_train, y_train, epochs=params['epochs'], validation_split=0.1,
                        verbose=1, batch_size=params['batch_size'], callbacks=tb)

    # save latest config
    if not isdir('./' + modelsfolder):
        mkdir('./' + modelsfolder)
    if not isdir('./' + modelsfolder + '/' + folder_name):
        mkdir('./' + modelsfolder + '/' + folder_name)

    model.save('./' + modelsfolder + '/' + folder_name)
    latest_file = open('./Models/Latest.txt', "w")
    latest_file.write('./' + modelsfolder + '/' + folder_name)
    latest_file.close()

    # save parameters
    # Remove to be able to save other parameters to json file
    params['model'] = ()
    params['policy_kwargs'] = ()
    p_file = json.dumps(params)
    f = open('./' + modelsfolder + '/' + folder_name + '/params.json', "w")
    f.write(p_file)
    f.close()

    # save hisotry
    h_file = json.dumps(history.history)
    f = open('./' + modelsfolder + '/' + folder_name + '/history.json', "w")
    f.write(h_file)
    f.close()

    print("Saved model to disk: " + str(params))
