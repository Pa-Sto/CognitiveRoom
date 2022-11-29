import random
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS
import networkx as nx
import json
from sklearn.decomposition import PCA
import io
import Datasets


def load_model_params(filename):
    # Identitfy
    filename_model_js = filename + "/model.json"
    filename_model_h5 = filename + "/weights.h5"
    filename_params = filename + "/params.json"

    # load model's parameters
    params_json = open(filename_params)
    params = json.load(params_json)

    # load and create model
    json_file = open(filename_model_js, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = None  # model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(filename_model_h5)
    print("Loaded model from disk: " + str(params))

    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    return params, loaded_model


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def create_m_matrix(prob_matrix, discount_factor=1., sequence_length=1):
    converge = False

    if not converge:
        m_matrix = np.zeros(prob_matrix.shape)
        for i in tqdm(range(sequence_length)):
            m_matrix += np.power(discount_factor, i) * np.linalg.matrix_power(prob_matrix, i)

    # deprecated
    else:
        diag = np.zeros(prob_matrix.shape)
        np.fill_diagonal(diag, 1)
        m_matrix = np.linalg.inv(diag - discount_factor * prob_matrix)

    # TODO normalize? why necessary?
    # m_matrix_norm = m_matrix / np.linalg.norm(m_matrix)
    for runner in range(m_matrix.shape[0]):
        m_matrix[runner, :] /= np.sum(m_matrix[runner, :])

    return m_matrix


def create_mds_matrix(SR, number_components=2):
    # dis = create_dissimilarity_matrix(SL)

    """
    inspired by
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """
    # print("creating mds-matrix with" + str(number_components) + " components...", end="")
    mds = MDS(n_components=number_components, n_init=100, max_iter=10000, eps=1e-5, n_jobs=-1)
    components = mds.fit_transform(SR)
    # print("done.")
    return components, mds


def create_pca(SR, number_components=2):
    """
    Inspired by:
    https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

    and by

    https://plotly.com/python/pca-visualization/
    """

    pca = PCA(n_components=number_components)
    return pca.fit_transform(SR)


def show_state_prob_graph(transition_probability_matrix, plot):
    # ------------------------------------------------------------------------------------------
    # for other styles: check https://networkx.org/documentation/stable/reference/drawing.html

    # maybe make nodes with node2vec
    # found in "Madjiheurem&Toni//State2vec: Off-Policy Successor Features Approximators"
    # ------------------------------------------------------------------------------------------
    # !!!!!Still very specialized to the language Model -- Needs adapation!!!!!
    # ------------------------------------------------------------------------------------------

    # Graph, Plot, Edges and weights init
    # figure = plt.figure(num="State Probability Graph")
    weights = []
    graph = nx.Graph()
    nodes_no = transition_probability_matrix.shape[0]

    adjectives = ['big', 'small', 'close', 'fast', 'slow', 'brave', 'difficult', 'good', 'bad', 'real']
    verbs = ['go', 'eat', 'walk', 'lift', 'drive', 'come', 'develope', 'think', 'hear', 'order']
    nouns = ['house', 'car', 'tree', 'rain', 'river', 'lamp', 'tent', 'dinner', 'night', 'day']
    pronouns = ['I', 'you', 'they', 'we']
    questions = ['do', 'can', 'why do', 'what do', 'will', 'must']

    word_groups = []
    word_groups.append(adjectives)
    word_groups.append(verbs)
    word_groups.append(nouns)
    word_groups.append(pronouns)
    word_groups.append(questions)

    word_label_coulors = [
        'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue',
        'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red',
        'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green',
        'orange', 'orange', 'orange', 'orange',
        'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']
    labels_l = word_groups
    labels = {}
    i = 0
    for l in labels_l:
        labels[i] = l
        i += 1
    for i in range(nodes_no):
        graph.add_node(i, color=word_label_coulors[i])
        # nodes.append(i)
        for j in range(nodes_no):
            if transition_probability_matrix[i, j] > 0:  # (and i < j )
                # edges.append((i, j))
                graph.add_edge(i, j, weights=transition_probability_matrix[i, j])
                weights.append(transition_probability_matrix[i, j])

    # Directed Graph which allows multiply edgeds between nodes

    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    my_pos = nx.spring_layout(graph, seed=100, iterations=1000)  # k=1./np.sqrt(np.max(visual.params.room_shape)))
    # my_pos = nx.kamada_kawai_layout(graph)

    # Plot erstellen und Graph anzeigen
    # figure_graph = plt.figure('State Probability Graph')
    node_color_dict = (graph.nodes(data='color'))
    node_color_dict = np.array(node_color_dict)[:, 1]
    nx.draw(graph, pos=my_pos, node_size=50, width=weights, node_color=node_color_dict, ax=plot)  # , with_labels=True)
    nx.draw_networkx_labels(graph, my_pos, labels, font_size=16, ax=plot)
    # figure.show()
    # plt.show()
    return graph


def hot_encoded_vector(size, state=None):
    np.random.seed(42)
    vector = np.zeros((size))
    if state is None:
        state = np.random.randint(0, size)

    vector[state] = 1

    return vector


def create_tpm_SL_AnimalDataSet(model):
    # Load Referencr Values and create tpm
    transition_probability_matrix_gt, animal_data, animal_names, animal_parameter, _ = Datasets.AnimalCognitiveRoom()
    transition_probability_matrix = np.zeros(transition_probability_matrix_gt.shape)
    # Iterate over all states and give their transition probabilities
    for i in range(transition_probability_matrix_gt.shape[0]):
        state = animal_data[i, :]
        state = state[np.newaxis, :]
        state_transition_prob = model.predict(state)
        transition_probability_matrix[i, :] = state_transition_prob

    return transition_probability_matrix


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))


def create_cdf_matrix(pdf):
    cdf_matrix = np.copy(pdf)

    for i in range(pdf.shape[0]):
        for j in range(1, pdf[i].size):  #
            cdf_matrix[i, j] += cdf_matrix[i, j - 1]

    return cdf_matrix


def get_sequences_AnimalDataSet(number_samples, alternate=0):
    # Starting sequence Builder
    print("creating sequences...")
    # Set Random Seed
    np.random.seed(420)
    # Load Data
    transition_probability_matrix, animal_data, animal_names, animal_parameter, animal_data_normalized = \
        Datasets.AnimalCognitiveRoom()

    x_train = np.zeros((number_samples, animal_data.shape[1]))
    y_train = np.zeros((number_samples, animal_data.shape[0]))

    cdf = create_cdf_matrix(transition_probability_matrix)
    # normal verteilte ausgangszustaende. muss evtl. angepasst werden?
    for i in tqdm(range(number_samples)):
        # startzustand
        state = random.randint(0, cdf.shape[0] - 1)

        if alternate == 0:
            x_train[i] = animal_data[state, :]
        else:
            animal_data_alternate = np.copy(animal_data[state, :])
            for j in range(4):
                factor = np.random.uniform(0, alternate) * animal_data_alternate[j]
                animal_data_alternate[j] += factor
            x_train[i] = animal_data_alternate
        # sample next state
        var = random.uniform(0, 1)
        # find corresponding state
        next_state = 0
        while cdf[state, next_state] < var:
            next_state += 1

        next_state = hot_encoded_vector(animal_data.shape[0], next_state)
        y_train[i] = next_state

    print("sequences done.")
    return x_train, y_train


def get_sequences_fromSR_AnimalDataSet(number_samples, t, df, alternate=0):
    # Starting sequence Builder
    print("creating sequences...")
    # Set Random Seed
    np.random.seed(420)
    # Load Data
    transition_probability_matrix, animal_data, animal_names, animal_parameter, animal_data_normalized = \
        Datasets.AnimalCognitiveRoom()
    # Calculate the SR Matrix for the TPM
    sr_matrix = create_m_matrix(transition_probability_matrix, df, t)

    x_train = np.zeros((number_samples, animal_data.shape[1]))
    y_train = np.zeros((number_samples, animal_data.shape[0]))

    cdf = create_cdf_matrix(sr_matrix)
    # normal verteilte ausgangszustaende. muss evtl. angepasst werden?
    for i in tqdm(range(number_samples)):
        # startzustand
        state = random.randint(0, cdf.shape[0] - 1)

        if alternate == 0:
            x_train[i] = animal_data[state, :]
        else:
            animal_data_alternate = np.copy(animal_data[state, :])
            for j in range(4):
                factor = np.random.uniform(0, alternate) * animal_data_alternate[j]
                animal_data_alternate[j] += factor
            x_train[i] = animal_data_alternate

        # sample next state
        var = random.uniform(0, 1)
        # find corresponding state
        next_state = 0
        while cdf[state, next_state] < var:
            next_state += 1

        next_state = hot_encoded_vector(animal_data.shape[0], next_state)
        y_train[i] = next_state

    print("sequences done.")
    return x_train, y_train


def gauss(img, mean=0, std=1):  # img is numpy array image
    gauss = np.random.normal(mean, std, img.size)  # mean=0, std=1
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2])
    return img + gauss


def set_random_missing_entries(test_data, missing_entries):
    # Copy array to scratch entries
    test_data_incomplete = np.copy(test_data)
    np.random.seed(20)
    index_list = []
    index = np.random.randint(-1, missing_entries)
    for i in range(missing_entries):
        if test_data_incomplete[0, index] != -1:
            test_data_incomplete[:, index] = -1
            index_list.append(index)

        while test_data_incomplete[0, index] == -1 and i < missing_entries - 1:
            index = np.random.randint(-1, missing_entries)

    return test_data_incomplete, index_list


def set_missing_entries_fixed(test_data, missing_entries, fixed_entry):
    # Copy array to scratch entries
    test_data_incomplete = np.copy(test_data)
    np.random.seed(42)
    index_list = []
    index = fixed_entry
    for i in range(missing_entries):
        if test_data_incomplete[0, index] != -1:
            test_data_incomplete[:, index] = -1
            index_list.append(index)

        while test_data_incomplete[0, index] == -1 and i < missing_entries:
            index = np.random.randint(-1, missing_entries)
    # print(test_data_incomplete)
    return test_data_incomplete, index_list


def set_missing_entries(test_data, missing_entries, protected_entry):
    # Copy array to scratch entries
    test_data_incomplete = np.copy(test_data)
    np.random.seed(42)
    index_list = []
    index = np.random.randint(-1, missing_entries)
    while index == protected_entry:
        index = np.random.randint(-1, missing_entries)
    for i in range(missing_entries):
        if test_data_incomplete[0, index] != -1:
            test_data_incomplete[:, index] = -1
            index_list.append(index)

        while test_data_incomplete[0, index] == -1 and i < missing_entries - 1 or index == protected_entry:
            index = np.random.randint(-1, missing_entries)
    # print(test_data_incomplete)
    return test_data_incomplete, index_list


def evaluate_model_on_cognitive_room_prediction(model, test_data, missing_entries, memory_matrix_training, metric,
                                                random=True, protected_entry=None, fixed_entry=None):
    # Set up test data with missing entries
    if random:
        test_data_incomplete, index_list = set_random_missing_entries(test_data, missing_entries)
    elif protected_entry == None:
        test_data_incomplete, index_list = set_missing_entries(test_data, missing_entries, protected_entry)
    else:
        test_data_incomplete, index_list = set_missing_entries_fixed(test_data, missing_entries, fixed_entry)
    # Use model to make prediciton on the test data
    trp_test_data = model.predict(test_data_incomplete)

    # Interpolate data based on memory matrix used for training
    interpolated_test_data = trp_test_data @ memory_matrix_training
    # Lets see how good we predicted depending on the metric we choose
    if metric == 'l2':
        difference_prediction_gt = test_data - interpolated_test_data
        euclidean_distande = np.sqrt(difference_prediction_gt * difference_prediction_gt)
        avg_distance_per_feature = np.sum(euclidean_distande, axis=0) / test_data.shape[0]
        avg_metric_per_feature = avg_distance_per_feature
    if metric == 'percentage':
        difference_prediction_gt = test_data - interpolated_test_data
        euclidean_distande = np.sqrt(difference_prediction_gt * difference_prediction_gt)
        perecentage_difference_per_feature = np.divide(100, np.max(test_data, axis=0)) * euclidean_distande
        avg_metric_per_feature = np.sum(perecentage_difference_per_feature, axis=0) / test_data.shape[0]

    return avg_metric_per_feature, euclidean_distande


def calculate_GDV(data_points, labels, number_classes):
    GDV = 0
    labels = np.array(labels)
    # Re-scale all data points
    data_points_scaled = np.copy(data_points)
    mean = np.mean(data_points_scaled)
    std = np.std(data_points_scaled)
    data_points_scaled = 0.5 * ((data_points_scaled - mean) / std)
    # Calculate mean intra-class distance for each class
    for i in range(number_classes):
        # index = np.where(labels==i)
        class_data1 = data_points_scaled[i == labels]
        if i != number_classes - 1:
            class_data2 = data_points_scaled[i + 1 == labels]
        inter_class_distance = 0
        intra_class_distance = 0
        for l in range(class_data1.shape[0] - 1):
            dp = np.sqrt(np.sum(np.square(class_data1[l] - class_data1[l + 1])))
            inter_class_distance += dp
            if i != number_classes - 1:
                for m in range(class_data2.shape[0]):
                    distance_intra_points = np.sqrt(np.sum(np.square(class_data1[l] - class_data2[m])))
                    intra_class_distance += distance_intra_points
        inter_class_distance = 2 * inter_class_distance / (class_data1.shape[0] * (class_data1.shape[0] - 1))
        if i != number_classes - 1:
            intra_class_distance = intra_class_distance / (class_data1.shape[0] * class_data2.shape[0])
        GDV += inter_class_distance / number_classes
        GDV -= 2 * intra_class_distance / (number_classes * (number_classes - 1))

    return GDV / np.sqrt(2)
