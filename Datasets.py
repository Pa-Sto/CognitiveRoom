import pandas as pd
import numpy as np


def AnimalCognitiveRoom(path='Dataset/AnimalCognitiveRoom.xlsx'):
    # Import Data from CSV file and convert to different numpy arrays
    raw_data = pd.read_excel(path, engine='openpyxl')
    animal_parameter = raw_data.loc[0, :].to_numpy()
    animal_names = raw_data['Animal Data Set'].to_numpy()
    animal_names = animal_names[1:33]
    animal_data = raw_data.to_numpy()
    animal_data = animal_data[1:33, 1:]

    # Creating TransitionProbability Matrix
    transition_probability_matrix = np.zeros((animal_data.shape[0], animal_data.shape[0]))
    # Normalizing data rows --> So big values exp. Height do not have higher influence than small values on distance metric
    animal_data_normalized = animal_data.astype(float)
    for i in range(animal_data.shape[1]):
        animal_data_normalized[:, i] = animal_data_normalized[:, i] / np.sum(animal_data_normalized[:, i])
    # Create transition probability Matrix based on euclidean distance of data distributions
    for row in range(transition_probability_matrix.shape[0]):
        for col in range(transition_probability_matrix.shape[1]):
            # Find transition for every animal and set transition to self to 0
            if row == col:
                transition_probability_matrix[row, col] = 0
            else:
                distance = np.sum(np.absolute(animal_data_normalized[row, :] - animal_data_normalized[col, :]))
                transition_probability_matrix[row, col] = 1 / (distance * distance)
    # Normalize probabilities
    for a in range(transition_probability_matrix.shape[0]):
        transition_probability_matrix[a, :] = transition_probability_matrix[a, :] \
                                              / np.sum(transition_probability_matrix[a, :])

    return transition_probability_matrix, animal_data.astype(
        float), animal_names, animal_parameter, animal_data_normalized


def AnimalCognitiveRoomTest(path='Dataset/AnimalCognitiveRoom.xlsx'):
    # Import Data from CSV file and convert to different numpy arrays
    raw_data = pd.read_excel(path, engine='openpyxl')
    animal_parameter = raw_data.loc[0, :].to_numpy()
    animal_names = raw_data['Animal Data Set'].to_numpy()
    animal_names = animal_names[33:40]
    animal_data = raw_data.to_numpy()
    animal_data = animal_data[33:40, 1:]

    # Creating TransitionProbability Matrix
    transition_probability_matrix = np.zeros((animal_data.shape[0], animal_data.shape[0]))
    # Normalizing data rows --> So big values exp. Height hae higher influence than small values on distance
    animal_data_normalized = animal_data.astype(float)
    for i in range(animal_data.shape[1]):
        animal_data_normalized[:, i] = animal_data_normalized[:, i] / np.sum(animal_data_normalized[:, i])
    # Create transition probability Matrix based on euclidean distance of data distributions
    for row in range(transition_probability_matrix.shape[0]):
        for col in range(transition_probability_matrix.shape[1]):
            # Find transition for every animal
            if row == col:
                transition_probability_matrix[row, col] = 0
            else:
                distance = np.sum(np.absolute(animal_data_normalized[row, :] - animal_data_normalized[col, :]))
                transition_probability_matrix[row, col] = 1 / (distance * distance)
    # Normalize probabilities
    for a in range(transition_probability_matrix.shape[0]):
        transition_probability_matrix[a, :] = transition_probability_matrix[a, :] \
                                              / np.sum(transition_probability_matrix[a, :])

    return transition_probability_matrix, animal_data.astype(
        float), animal_names, animal_parameter, animal_data_normalized
