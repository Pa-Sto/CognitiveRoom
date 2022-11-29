#import Datasets
import numpy as np
import matplotlib.pyplot as plt
#from keras.models import load_model
import helper_functions
import train
import os
import spacy

print('Welcome to the Playground, may testing be successful!')

# transition_probability_matrix, transition_probability_matrix_test, train, test = Datasets.Cifar10(True)
# state_prob_matrix, vector_list, word_list = Datasets.Ogden_2000_list()
# matrix, vectors, data = Datasets.WordVectors_Three_Domains()
# model = load_model('./Models/2022.11.09-12:22:34(WordVec_Epochs:1000)')
#
# spacy.require_cpu()
# nlp = spacy.load("en_core_web_lg")
#
# test_word = nlp('weed')
# test_vector = test_word.vector
# test_predict = model.predict(test_vector[np.newaxis])
# matrix = np.vstack((matrix, test_predict))





# mds1, mds1_calc = helper_functions.create_mds_matrix(state_prob_matrix)
#
# for i in range(state_prob_matrix.shape[0]):
#         plt.scatter(mds1[i, 0], mds1[i, 1], c='orange')
#         #plt.annotate(word_list[i],(mds1[i, 0], mds1[i, 1]))


# label1 = True
# label2 = True
# label3 = True
# for i in range (91):
#     if i < 30:
#         if label1 is True:
#             plt.scatter(mds1[i,0], mds1[i,1], c='blue', label='Animlas')
#             label1 = False
#         else:
#             plt.scatter(mds1[i, 0], mds1[i, 1], c='blue')
#     if i > 30 and i < 60:
#         if label2 is True:
#             plt.scatter(mds1[i,0], mds1[i,1], c='green', label='Vehicels')
#             label2 = False
#         else:
#             plt.scatter(mds1[i, 0], mds1[i, 1], c='green')
#     if i > 60 and i < 90:
#         if label3 is True:
#             plt.scatter(mds1[i, 0], mds1[i, 1], c='red', label='Furniture')
#             label3 = False
#         else:
#             plt.scatter(mds1[i, 0], mds1[i, 1], c='red')
#     if i >= 90:
#         plt.scatter(mds1[i, 0], mds1[i, 1], c='black')
#         plt.annotate(test_word,(mds1[i, 0], mds1[i, 1]))

# plt.legend()
# plt.show()

''' 
transition_probability_matrix, animal_data, animal_names, animal_parameter, animal_data_normalized = Datasets.AnimalCognitiveRoom()
_, animal_data_test, animal_names_test, _ ,\
_ = Datasets.AnimalCognitiveRoom(path='/Users/paul/Documents/Promotion/Data/AnimalDataSets/AnimalCognitiveRoomTest.xlsx')

m_test = helper_functions.create_m_matrix(transition_probability_matrix, 1, 10)

animal_data_test = animal_data_test[np.newaxis, 0]
############### MDS - Missing vs Interpolation ###########################
model10 =  load_model('Models/AnimalPredictionJaguarDF10')
model07 =  load_model('Models/AnimalPredictionJaguarDF07')
model03 =  load_model('Models/AnimalPredictionJaguarDF03')
gt_list = []
gt_list.append(animal_data_test)
list03 = []
list03.append(animal_data_test)
list07 = []
list07.append(animal_data_test)
list10 = []
list10.append(animal_data_test)
for i in range(1,7):
    incomplete_data_test,_ = helper_functions.set_random_missing_entries(animal_data_test,i)
    gt_list.append(incomplete_data_test)
    prediction03 = model03.predict(incomplete_data_test)
    prediction03 = prediction03@animal_data
    list03.append(prediction03)
    prediction07 = model07.predict(incomplete_data_test)
    prediction07 = prediction07@animal_data
    list07.append(prediction07)
    prediction10 = model10.predict(incomplete_data_test)
    prediction10 = prediction10@animal_data
    list10.append(prediction10)
#Reformat and normalize
new_shape = animal_data_test.shape[0]*7
test_data_max = np.max(animal_data_test, axis=0)
gt_list = np.array(gt_list)
gt_list = np.reshape(gt_list,((animal_data_test,7)))
list10 = np.array(list10).reshape((animal_data_test,7))
list07 = np.array(list07).reshape((animal_data_test,7))
list03 = np.array(list03).reshape((animal_data_test,7))
gt_list = np.divide(gt_list,test_data_max)
list03 = np.divide(list03,test_data_max)
list07 = np.divide(list07,test_data_max)
list10 = np.divide(list10,test_data_max)
mds_gt,_ = helper_functions.create_mds_matrix(np.array(gt_list))
mds03,_ = helper_functions.create_mds_matrix(np.array(list03))
mds07,_ = helper_functions.create_mds_matrix(np.array(list07))
mds10,_ = helper_functions.create_mds_matrix(np.array(list10))

word_label_coulors = ['black','blue', 'red','green', 'orange', 'pink', 'purple']
number_missing_features_Labels = ['Ground Truth','1', '2', '3', '4', '5', '6']
fig, axs = plt.subplots(1,4, figsize=(30,7))


c = 0
for i in range(new_shape):
    if i<6:
        axs[0].annotate(animal_names_test[i], (mds_gt[i, 0], mds_gt[i, 1]), fontsize=15)
        axs[1].annotate(animal_names_test[i], (mds03[i, 0], mds03[i, 1]), fontsize=15)
        axs[2].annotate(animal_names_test[i], (mds07[i, 0], mds07[i, 1]), fontsize=15)
        axs[3].annotate(animal_names_test[i], (mds10[i, 0], mds10[i, 1]), fontsize=15)

    axs[0].scatter(mds_gt[i, 0], mds_gt[i, 1], c=word_label_coulors[c], s=100)
    axs[1].scatter(mds03[i, 0], mds03[i, 1], c=word_label_coulors[c],s=100)
    axs[2].scatter(mds07[i, 0], mds07[i, 1], c=word_label_coulors[c],s=100)
    axs[3].scatter(mds10[i, 0], mds10[i, 1], c=word_label_coulors[c], s=100)

    if i%6 == 0 and i != 0:
        axs[0].scatter(mds_gt[i, 0], mds_gt[i, 1], c=word_label_coulors[c], label=number_missing_features_Labels[c])
        c+=1
    if  i%6 == 0 and c==6:
        axs[0].scatter(mds_gt[i, 0], mds_gt[i, 1], c=word_label_coulors[c], label=number_missing_features_Labels[c])


axs[0].legend(bbox_to_anchor=(0, 1.02), loc=0, title ='Number of Missing Features', fontsize=10)
axs[0].set_title('Ground Truth', fontsize=15)
axs[1].set_title('Interpolation with df=0.3', fontsize=15)
axs[2].set_title('Interpolation with df=0.7', fontsize=15)
axs[3].set_title('Interpolation with df=1.0', fontsize=15)
plt.show()
'''
'''
############### Interpolation Graph ###########################
#Creating Error Prediction Plot
numpy_averages = np.zeros((7,7))
numpy_max = np.zeros((7,7))
numpy_min = np.zeros((7,7))
numpy_min = 100
#for n in range(10):
for model_name in os.listdir("Models/InterpolationGraph"):
    #train.main()
    _file = ""
    #latest_filename_filename = open("./Models/Latest.txt", "r")
    #latest_filename_filename = open("Models/InterpolationGraph/" + str(model_name), "r")
    #model_files = latest_filename_filename.read()
    model_files = "Models/InterpolationGraph/" + str(model_name)
    model = load_model(model_files)
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
    numpy_average = np.array(averages)
    numpy_max = np.maximum(numpy_max,numpy_average)
    numpy_min = np.minimum(numpy_min,numpy_average)
    numpy_averages += numpy_average

numpy_averages /= 10
bars = {0:'bar0',1:'bar1',2:'bar2',3:'bar3',4:'bar4',5:'bar5',6:'bar6',}
bar_width = 0.13
#bar_color =['blue','orange','green','red','lila','brown','pink']
error_params = {'lw':1.5}
fig, axs = plt.subplots(1, 1, figsize=(25,10))
for p in range(0,7):
    x = np.arange(0,len(numpy_averages[p]))
    error = np.vstack((numpy_min[p], numpy_max[p]))
    bars[p] = axs.bar(x + p*bar_width,numpy_averages[p], yerr= error, width=bar_width, error_kw=error_params)
    bars[p] = axs.plot(numpy_averages[p], lw=3)
#plt.title('Distance of Prediction to GT Feature in % \n in comparison to number of missing feature inputs ')
plt.ylabel('Distance of Prediction to GT Feature in %', fontsize=25)
plt.xlabel('Number of missing features', fontsize=25)
plt.legend(['Height', 'Weight', '#Legs', 'Danger', 'Reprodcution', 'Fur', 'Lungs'], loc='upper left',fontsize=25)
plt.tick_params(labelsize=20)
plt.show()

'''



"""
_, animal_data_test, animal_names_test, _ ,\
_ = Datasets.AnimalCognitiveRoom(path='/Users/paul/Documents/Promotion/Data/AnimalDataSets/AnimalCognitiveRoomTest.xlsx')
jaguar_sparse = np.copy(animal_data_test[0])
jaguar_sparse[3] = -1
jaguar_sparse[6] = -1
jaguar_sparse[5] = -1
jaguar_sparse = jaguar_sparse[np.newaxis,:]
print('Leopard Test: ' + str(animal_data_test[0]))
print('Leopard Test Sparse: ' + str(jaguar_sparse))
model10 =  load_model('Models/AnimalPredictionJaguarDF10')
pred1 = model10.predict(jaguar_sparse)@animal_data
print('Leopard Test Predicition 1.0: ' + str(pred1))
model07 =  load_model('Models/AnimalPredictionJaguarDF07')
pred2 = model07.predict(jaguar_sparse)@animal_data
print('Leopard Test Predicition 0.7: ' + str(pred2))
model03 =  load_model('Models/AnimalPredictionJaguarDF03')
pred3 = model03.predict(jaguar_sparse)@animal_data
print('Leopard Test Predicition 0.3: ' + str(pred3))
"""

'''
transition_probability_matrix, train_images, test_images = Datasets.MNIST_ImageSet()
sr_matrix = helper_functions.create_m_matrix(transition_probability_matrix,0.3,10)
state = transition_probability_matrix[0]
state = state[None,:]
state[0] = None
train_images_edit = train_images.reshape(train_images.shape[0],-1)

test_image = sr_matrix[8]@train_images_edit

test_image = test_image.reshape(train_images[0].shape)

plot, axs = plt.subplots(2)
axs[0].imshow(train_images[8])
axs[1].imshow(test_image)
plt.show()
'''
'''
transition_probability_matrix, animal_data, animal_names, animal_parameter, animal_data_normalized = Datasets.AnimalCognitiveRoom()

_, animal_data_test, animal_names_test, _ ,\
_ = Datasets.AnimalCognitiveRoom(path='/Users/paul/Documents/Promotion/Data/AnimalDataSets/AnimalCognitiveRoomTest.xlsx')
#test = helper_functions.test_feature_predictions(transition_probability_matrix, animal_data, 5)
#SR-t=10 Model
#model =  load_model('./Models/2022.05.20-10:10:12(AnimalRoom_Epochs:1000)')
#TPM MODEL
model =  load_model('./Models/2022.05.16-09:12:10(AnimalRoom_Epochs:1000)')
average = []
averages = []
for i in range(0,7):
    avg_distance_per_feature, euclidean_distande = \
        helper_functions.evaluate_model_on_cognitive_room_prediction(model, animal_data_test,i,animal_data, 'percentage')
    average.append(np.average(avg_distance_per_feature))
    averages.append(avg_distance_per_feature.T)
plt.plot(averages)


#fig, axs = plt.subplots(1,1)
#axs.bar(animal_parameter[1:],avg_distance_per_feature)
#axs.tick_params(labelsize=10)
plt.title('Distance of Prediction to GT Feature in % \n in comparison to number of missing feature inputs ')
plt.ylabel('Distance of Prediction to GT Feature in %')
plt.xlabel('Number of missing features')
plt.legend(['Height', 'Weight','#Legs','Danger','Reprodcution','Fur','Lungs'], loc='upper left')
plt.show()


prediction = model.predict(animal_data_test)
prediction_data = prediction@animal_data
differnece = animal_data_test-prediction_data
print('Prediction:' +str(prediction_data[0,:]))
print('Ground Truth:'+ str(animal_data_test[0,:]))
'''
# #eigenvalues, eigenvectors = np.linalg.eig(transition_probability_matrix)
# sr0 = helper_functions.create_m_matrix(transition_probability_matrix,.3,10)
# sr1 = helper_functions.create_m_matrix(transition_probability_matrix,.7,10)
# sr2 = helper_functions.create_m_matrix(transition_probability_matrix,1,10)
'''
fig, axs = plt.subplots(1,3, figsize=(15,5))
axs[0].imshow(sr0)
axs[0].set_title('d=0.3')
axs[1].imshow(sr1)
axs[1].set_title('t=0.7')
axs[2].imshow(sr2)
axs[2].set_title('t=1.0')
'''
# #0 - mammals, 1 - insects, 2 - amphibiens
# labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2]
#
# mds1, mds1_calc = helper_functions.create_mds_matrix(sr0)
# mds2, mds2_calc = helper_functions.create_mds_matrix(sr1)
# mds3, mds3_calc = helper_functions.create_mds_matrix(sr2)
#
# gdv1 = helper_functions.calculate_GDV(mds1,labels,3)
# gdv2 = helper_functions.calculate_GDV(mds2,labels,3)
# gdv3 = helper_functions.calculate_GDV(mds3,labels,3)
# print('GDV of MDS1,df:0.3: '+ str(gdv1))
# print('GDV of MDS2,df:0.7: '+ str(gdv2))
# print('GDV of MDS3,df:1.0: '+ str(gdv3))
# #Reference Anmimals
#
# mammal = np.array([50,50,4,10,0,1,1])
# insect = np.array([3,0.005,6,0,1,0,0])
# amphibian = np.array([10,0.05,4,0,1,0,1])
#
# #Color Stuff
# c = np.arange(0,transition_probability_matrix.shape[0],1)
# fig2, axs2 = plt.subplots(1,3, figsize=(16,4.5))
# axs2[0].scatter(mds1[:,0], mds1[:,1],c = c, cmap='plasma')
# axs2[0].set_title('t=10, df=0.3', fontsize=15)
# for j, label in enumerate(animal_names):
#     axs2[0].annotate(label, (mds1[j, 0], mds1[j, 1]))
# axs2[1].scatter(mds2[:,0], mds2[:,1],c=c,cmap='plasma')
# axs2[1].set_title('t=10, df=0.7', fontsize=15)
# for j, label in enumerate(animal_names):
#     axs2[1].annotate(label, (mds2[j, 0], mds2[j, 1]))
# axs2[2].scatter(mds3[:,0], mds3[:,1],c=c,cmap='plasma')
# axs2[2].set_title('t=10, df=1.0', fontsize=15)
# for j, label in enumerate(animal_names):
#     axs2[2].annotate(label, (mds3[j, 0], mds3[j, 1]))
#
# plt.show()

'''
m_matrix = helper_functions.create_m_matrix(transition_probability_matrix,discount_factor=1,sequence_length=3)
tpm_test = np.power(transition_probability_matrix[0,:],3)
pander = np.array([-1,-1,-1,100,-1,-1,-1])
pander = pander.reshape((1,7))
model = load_model('./Models/2022.01.24-09:09:23(AnimalRoom_Epochs:100)')
sr_pander = model.predict(pander)
pander_inter = sr_pander@animal_data
print(pander_inter)


test1 = eigenvectors@animal_data_normalized
test2 = eigenvectors@animal_data
test3 = eigenvectors[:,0]@animal_data_normalized
test4 = eigenvectors[0,:]@animal_data_normalized

model = load_model('./Models/2022.01.24-09:09:23(AnimalRoom_Epochs:100)')

test_vector = np.array([350,60000,4,60,0,0,1])

test_vector = test_vector.reshape((1,7))

sr_test = model.predict(test_vector)

interpolated_vector = sr_test@animal_data
'''
print('Testing Done')