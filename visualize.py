import json
import random
import Datasets
import helper_functions
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

# ------------------------------------------------------------------------
# What to show??
prediction = True
ground_truth = False
m_matrix = False
mds = True
pca = False
accuracy = False
loss = False
evaluate_interpolation = False
plot_list = [prediction,m_matrix, mds, pca, ground_truth, accuracy, loss, evaluate_interpolation]


# -------------------------------------------------------------------------

def main(model_files=None):
    global m_matrix, mds, pca, plot_list
    # Init Variable for plot count
    plot_count = 0
    # collect paths - if None pick last model which got trained
    if model_files is None:  # no args
        _file = ""
        latest_filename_filename = open("./Models/Latest.txt", "r")
        model_files = latest_filename_filename.read()

    print("Identified:" + model_files)
    # Load Model data from text file
    model = load_model(model_files)
    #####Get Params
    filename_params = model_files + "/params.json"
    # load model's parameters
    params_json = open(filename_params)
    params = json.load(params_json)
    ################ Choosing Model ##############################################
    #### Animal Data
    if params['model_name'] == 'AnimalRoom':
        transition_probability_matrix_gt, animal_data, animal_names, animal_parameter, _ = Datasets.AnimalCognitiveRoom()
        # Make preditctions for all states with the model
        transition_probability_matrix = helper_functions.create_tpm_SL_AnimalDataSet(model)
        _, animal_data_test, animal_names_test, _, \
        _ = Datasets.AnimalCognitiveRoom(
            path='Dataset/AnimalCognitiveRoomTest.xlsx')
        if params['SR']:
            transition_probability_matrix_gt = helper_functions.create_m_matrix(transition_probability_matrix_gt, params['df'],params['t'])
    ##############################################################################
    # Initialize plots
    plot_number = np.sum(plot_list)
    fig, axs = plt.subplots(1, plot_number, figsize=(25,10))
    # Setting Font Size of Axis and Labels
    plt.rc('font', size=20)
    plt.rc('axes', labelsize=20)
    if prediction:
        # Show Prediction Vector
        axs[plot_count].imshow(transition_probability_matrix)
        axs[plot_count].set_title('Transition Probability Matrix')
        axs[plot_count].tick_params(labelsize=20)
        plot_count += 1
    if params['SR']:
        fig.suptitle('Parameter: t=' + str(params['t']) + ', df=' + str(params['df']))
    if ground_truth:
        axs[plot_count].imshow(transition_probability_matrix_gt)
        axs[plot_count].set_title('Transition Probability Matrix Ground Truth')
        axs[plot_count].tick_params(labelsize=20)
        plot_count += 1
    if m_matrix:
        m_matrix = helper_functions.create_m_matrix(transition_probability_matrix, discount_factor=params['df'],
                                                   sequence_length=params['t'])
        axs[plot_count].imshow(m_matrix)
        axs[plot_count].set_title('SR, t=' + str(params['t']) + 'DF=' + str(params['df']))
        plot_count += 1
    if mds:
        # You need to define if use transition_probability_matrix or m_matrix
        mds,_ = helper_functions.create_mds_matrix(transition_probability_matrix)
        c = np.arange(0, transition_probability_matrix.shape[0], 1)
        axs[plot_count].scatter(mds[:, 0], mds[:, 1], c = c, cmap='plasma')
        axs[plot_count].set_title('MDS')
        if params['model_name'] == 'AnimalRoom':
            for j, label in enumerate(animal_names):
                axs[plot_count].annotate(label, (mds[j, 0], mds[j, 1]))
        axs[plot_count].tick_params(labelsize=20)
        plot_count += 1
    if pca:
        # You need to define if use transition_probability_matrix or m_matrix
        pca = helper_functions.create_pca(transition_probability_matrix)
        axs[plot_count].scatter(pca[:, 0], pca[:, 1])
        axs[plot_count].set_title('PCA')
        plot_count += 1
    if accuracy:
        filename_history = model_files + "/history.json"
        # load model's parameters
        params_json = open(filename_history)
        history = json.load(params_json)
        axs[plot_count].plot(history['accuracy'])
        axs[plot_count].plot(history['val_accuracy'])
        axs[plot_count].set_title('Model Accuracy')
        axs[plot_count].set_ylim(0,1)
        axs[plot_count].set_ylabel('Accuracy', fontsize=20)
        axs[plot_count].set_xlabel('Epoch', fontsize=20)
        axs[plot_count].legend(['train', 'val'], loc='upper left')
        axs[plot_count].tick_params(labelsize=20)
        plot_count += 1
    if loss:
        filename_history = model_files + "/history.json"
        # load model's parameters
        params_json = open(filename_history)
        history = json.load(params_json)
        axs[plot_count].plot(history['loss'])
        axs[plot_count].plot(history['val_loss'])
        axs[plot_count].set_title('Model Loss')
        axs[plot_count].set_ylabel('Loss', fontsize=20)
        axs[plot_count].set_xlabel('Epoch', fontsize=20)
        axs[plot_count].legend(['train', 'val'], loc='upper left')
        axs[plot_count].tick_params(labelsize=20)
        plot_count += 1
    if evaluate_interpolation:
        #AnimalRoom Specific
        average = []
        averages = []
        for j in range(0,7):
            for i in range(0, 7):
                avg_distance_per_feature, euclidean_distande = \
                    helper_functions.evaluate_model_on_cognitive_room_prediction(model, animal_data_test, i, animal_data,
                                                                             'percentage', random=False,protected_entry=None, fixed_entry=j)
                average.append(avg_distance_per_feature[j])
            averages.append(average)
            average = []
        for p in range(0,7):
            axs[plot_count].plot(averages[p])
        axs[plot_count].set_title('Distance of Prediction to GT Feature in % \n in comparison to number of missing feature inputs ')
        axs[plot_count].set_ylabel('Distance of Prediction to GT Feature in %')
        axs[plot_count].set_xlabel('Number of missing features')
        axs[plot_count].legend(['Height', 'Weight', '#Legs', 'Danger', 'Reprodcution', 'Fur', 'Lungs'], loc='upper left')
        plot_count += 1
    plt.show()


    print('RSME:' +str(np.average(np.sqrt((transition_probability_matrix-transition_probability_matrix_gt)*(transition_probability_matrix-transition_probability_matrix_gt)))))
