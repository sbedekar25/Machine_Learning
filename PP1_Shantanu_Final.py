import numpy as np
import matplotlib.pyplot as plt
import collections


with open('training_data.txt') as f:
    training_data = [word for line in f for word in line.split()]

with open('test_data.txt') as f:
    test_data = [word for line in f for word in line.split()]

# number of words in the entire training data
N = len(training_data)

#Word dictionary from train and test data

vocabulary = training_data + test_data

list_set1 = set(vocabulary)
distinct_words = (list(list_set1))
K = len(distinct_words)
print(K)


# Task 1: Model Training, Prediction and Evaluation

ALPHA_PRIME = 5
ALPHA_0 = ALPHA_PRIME * 1.0 * K


TRAINING_SETS = [N / 128, N / 64, N / 16, N / 4, N] # training  sizes


PERPLEXITY_MLE_TRAIN_ALL_SETS = []
PERPLEXITY_MAP_TRAIN_ALL_SETS = []
PERPLEXITY_PD_TRAIN_ALL_SETS = []
PERPLEXITY_MLE_TEST_ALL_SETS = []
PERPLEXITY_MAP_TEST_ALL_SETS = []
PERPLEXITY_PD_TEST_ALL_SETS =  []

#Run loop for each training data size
for i in range(len(TRAINING_SETS)):

    N1 = int(TRAINING_SETS[i])
    # dictionary of word frequency in training data
    training_dic = {}

    for j in range(N1):
        training_dic[training_data[j]] = training_dic.get(training_data[j], 0.0) + 1.0

    p_ML = {}
    p_MAP = {}
    p_pd = {}
    for k in range(K):
        m_k = training_dic.get(distinct_words[k], 0.01)
        alpha_k = ALPHA_PRIME
        p_ML [distinct_words[k]] =  m_k / N1

        p_MAP[distinct_words[k]] = (m_k + alpha_k - 1) / (N1 + ALPHA_0 - K)
        p_pd [distinct_words[k]] = (m_k + alpha_k) / (N1 + ALPHA_0)

    # Training perplexity for different models
    print(p_ML[distinct_words[k]])
    PERPLEXITY_MLE_TRAINING = 0.0
    PERPLEXITY_MAP_TRAINING = 0.0
    PERPLEXITY_PD_TRAINING  = 0.0

    for j in range(N1):
        PERPLEXITY_MLE_TRAINING = PERPLEXITY_MLE_TRAINING + np.log(p_ML[training_data[j]])
        PERPLEXITY_MAP_TRAINING = PERPLEXITY_MAP_TRAINING + np.log(p_MAP[training_data[j]])
        PERPLEXITY_PD_TRAINING  = PERPLEXITY_PD_TRAINING  + np.log(p_pd[training_data[j]])
    C=(-1.0) / N1
    PERPLEXITY_MLE_TRAINING = np.exp(C*PERPLEXITY_MLE_TRAINING)
    PERPLEXITY_MAP_TRAINING = np.exp(C*PERPLEXITY_MAP_TRAINING)
    PERPLEXITY_PD_TRAINING =  np.exp(C*PERPLEXITY_PD_TRAINING)

    PERPLEXITY_MLE_TRAIN_ALL_SETS.append(PERPLEXITY_MLE_TRAINING)
    PERPLEXITY_MAP_TRAIN_ALL_SETS.append(PERPLEXITY_MAP_TRAINING)
    PERPLEXITY_PD_TRAIN_ALL_SETS.append(PERPLEXITY_PD_TRAINING)

    # Test perplexity for differrent models

    PERPLEXITY_MLE_TEST = 0.0
    PERPLEXITY_MAP_TEST = 0.0
    PERPLEXITY_PD_TEST  = 0.0
    N_test_data= len(test_data)
    for j in range(N_test_data):
        PERPLEXITY_MLE_TEST = PERPLEXITY_MLE_TEST + np.log(p_ML[test_data[j]])
        PERPLEXITY_MAP_TEST = PERPLEXITY_MAP_TEST + np.log(p_MAP[test_data[j]])
        PERPLEXITY_PD_TEST  = PERPLEXITY_PD_TEST  + np.log(p_pd[test_data[j]])
    C1=(-1.0) / N_test_data
    PERPLEXITY_MLE_TEST = np.exp(C1 * PERPLEXITY_MLE_TEST)
    PERPLEXITY_MAP_TEST = np.exp(C1 * PERPLEXITY_MAP_TEST)
    PERPLEXITY_PD_TEST =  np.exp(C1 * PERPLEXITY_PD_TEST)

    PERPLEXITY_MLE_TEST_ALL_SETS.append(PERPLEXITY_MLE_TEST)
    PERPLEXITY_MAP_TEST_ALL_SETS.append(PERPLEXITY_MAP_TEST)
    PERPLEXITY_PD_TEST_ALL_SETS.append(PERPLEXITY_PD_TEST)



print("Task I: Model Training, Prediction and Evaluation              ")
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print("PERPLEXITY MLE TRAINING")
print("Perplexities on training data N/128 -> ",PERPLEXITY_MLE_TRAIN_ALL_SETS[0])
print("Perplexities on training data N/64  -> ",PERPLEXITY_MLE_TRAIN_ALL_SETS[1])
print("Perplexities on training data N/16  -> ",PERPLEXITY_MLE_TRAIN_ALL_SETS[2])
print("Perplexities on training data N/4   -> ",PERPLEXITY_MLE_TRAIN_ALL_SETS[3])
print("Perplexities on training data N     -> ",PERPLEXITY_MLE_TRAIN_ALL_SETS[4])

print("----------------------------------------------------------------")

print("PERPLEXITY MLE TEST(TEST Data has 640000  words)")
print("Perplexities on test using  training data N/128  -> ",PERPLEXITY_MLE_TEST_ALL_SETS[0])
print("Perplexities on test using  training data N/64   -> ",PERPLEXITY_MLE_TEST_ALL_SETS[1])
print("Perplexities on test using  training data N/16   -> ",PERPLEXITY_MLE_TEST_ALL_SETS[2])
print("Perplexities on test using  training data N/4    -> ",PERPLEXITY_MLE_TEST_ALL_SETS[3])
print("Perplexities on test using  training data N      -> ",PERPLEXITY_MLE_TEST_ALL_SETS[4])

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print("PERPLEXITY MAP TRAINING")
print("Perplexities on training data N/128  -> ",PERPLEXITY_MAP_TRAIN_ALL_SETS[0])
print("Perplexities on training data N/64   -> ",PERPLEXITY_MAP_TRAIN_ALL_SETS[1])
print("Perplexities on training data N/16   -> ",PERPLEXITY_MAP_TRAIN_ALL_SETS[2])
print("Perplexities on training data N/4    -> ",PERPLEXITY_MAP_TRAIN_ALL_SETS[3])
print("Perplexities on training data N      -> ",PERPLEXITY_MAP_TRAIN_ALL_SETS[4])

print("----------------------------------------------------------------")

print("PERPLEXITY MAP TEST (TEST Data has 640000  words)")
print("Perplexities on test using  training data N/128 -> ",PERPLEXITY_MAP_TEST_ALL_SETS[0])
print("Perplexities on test using  training data N/64  -> ",PERPLEXITY_MAP_TEST_ALL_SETS[1])
print("Perplexities on test using  training data N/16  -> ",PERPLEXITY_MAP_TEST_ALL_SETS[2])
print("Perplexities on test using  training data N/4   -> ",PERPLEXITY_MAP_TEST_ALL_SETS[3])
print("Perplexities on test using  training data N     -> " ,PERPLEXITY_MAP_TEST_ALL_SETS[4])

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")

print("PERPLEXITY PD TRAINING")
print("Perplexities on training data N/128 -> ",PERPLEXITY_PD_TRAIN_ALL_SETS[0])
print("Perplexities on training data N/64  -> ",PERPLEXITY_PD_TRAIN_ALL_SETS[1])
print("Perplexities on training data N/16  -> ",PERPLEXITY_PD_TRAIN_ALL_SETS[2])
print("Perplexities on training data N/4   -> ",PERPLEXITY_PD_TRAIN_ALL_SETS[3])
print("Perplexities on training data N     -> ",PERPLEXITY_PD_TRAIN_ALL_SETS[4])


print("----------------------------------------------------------------")
print("PERPLEXITY PD TEST (TEST Data has 640000  words)")
print("Perplexities on test  using  training data N/128 -> ",PERPLEXITY_PD_TEST_ALL_SETS[0])
print("Perplexities on test  using  training  data N/64  -> ",PERPLEXITY_PD_TEST_ALL_SETS[1])
print("Perplexities on test using  training   data N/16  -> ",PERPLEXITY_PD_TEST_ALL_SETS[2])
print("Perplexities on test  using  training  data N/4   -> ",PERPLEXITY_PD_TEST_ALL_SETS[3])
print("Perplexities on test using  training  data N     -> ",PERPLEXITY_PD_TEST_ALL_SETS[4])

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
#Perplexity Plots

plt.plot(TRAINING_SETS, PERPLEXITY_MLE_TRAIN_ALL_SETS,'mv--',label='MLE_train')
plt.plot(TRAINING_SETS, PERPLEXITY_MLE_TEST_ALL_SETS,'m^--' ,label='MLE_test')
plt.plot(TRAINING_SETS, PERPLEXITY_MAP_TRAIN_ALL_SETS,'cD-',label='MAP_train')
plt.plot(TRAINING_SETS, PERPLEXITY_MAP_TEST_ALL_SETS,'cx--',label='MAP_test')
plt.plot(TRAINING_SETS, PERPLEXITY_PD_TRAIN_ALL_SETS,'y*-' ,label='PD_train')
plt.plot(TRAINING_SETS, PERPLEXITY_PD_TEST_ALL_SETS,'y8--' ,label='PD_test')

plt.xlabel('N(training data size)')
plt.ylabel('Perplexities')
plt.title('The Perplexities on the train and test data under MLE,MAP And PD')

plt.legend()
plt.grid()
plt.show()


# Task 2: Model Selection


ALPHA_PRIME_list = range(1, 11, 1) #alpha parameter range
N1 = N / 128 #training size

PERPLEXITY_PD_TEST_ALL_SETS = []

log_evidence = []
N1=int(N1)
for ALPHA_PRIME in ALPHA_PRIME_list:
    ALPHA_0 = K * ALPHA_PRIME

    #log evidence on training data

    temp_log_evidence = 0.0

    for k in range(N1):
        temp_log_evidence = temp_log_evidence + (-1.0) * np.log(ALPHA_0 + k)

    training_dic = {}

    for j in range(N1):
        training_dic[training_data[j]] = training_dic.get(training_data[j], 0) + 1

    #Perplexity on test data and log evidence on training data

    p_pd = {}

    for k in range(K):
        m_k = training_dic.get(distinct_words[k], 0.01)
        alpha_k = ALPHA_PRIME
        p_pd[distinct_words[k]] = (m_k + alpha_k) * 1.0 / (N1 + ALPHA_0)
        if (m_k >= 1):

            for i in range(m_k):
                temp_log_evidence += np.log(alpha_k + i)

    PERPLEXITY_PD_TEST = 0.0

    for j in range(len(test_data)):
        PERPLEXITY_PD_TEST = PERPLEXITY_PD_TEST + np.log(p_pd[test_data[j]])

    PERPLEXITY_PD_TEST = np.exp(PERPLEXITY_PD_TEST * (-1.0) / len(test_data))
    PERPLEXITY_PD_TEST_ALL_SETS.append(PERPLEXITY_PD_TEST)

    log_evidence.append(temp_log_evidence)

PERPLEXITY_PD_TEST_ALL_SETS = [(int)(item) for item in PERPLEXITY_PD_TEST_ALL_SETS]
log_evidence = [(int)(item) for item in log_evidence]



print("----------------------------------------------------------------")
print("Task II: Model Selection                                       ")
print("----------------------------------------------------------------")
print("The Perplexities on test set for ALPHA_PRIME = 1.0, ...., 10.0 ")
for k in range(10):
      print("Alpha ", k+1 ,"  Perplexity ",PERPLEXITY_PD_TEST_ALL_SETS[k])

print(PERPLEXITY_PD_TEST_ALL_SETS)
for k in range(10):
      print("log evidence ", k+1 ,"  ",log_evidence[k])


#Task 2 plots

#Task 2 plots

plt.figure(1)
plt.subplot(121)
plt.plot(ALPHA_PRIME_list, PERPLEXITY_PD_TEST_ALL_SETS,'r*-')
plt.xlabel('alpha prime')
plt.ylabel('Perplexities on test data')
plt.title('The Perplexities on the test data')
plt.grid()
plt.subplot(122)
plt.plot(ALPHA_PRIME_list, log_evidence,'b*-')
plt.xlabel('alpha prime')
plt.ylabel('log evidence on training data')
plt.title('log evidence ')
plt.grid()


plt.show()


# Task 3: Author Identification

full_dic = {}
training_dic = {}

#Traing data pg121 reading
with open('pg121.txt.clean') as f:
    training = [word for line in f for word in line.split()]
word_c_dict =collections.Counter(training)
training_dic=dict(word_c_dict)


#Reading test data in pg141
with open('pg141.txt.clean') as f:
    pg141 = [word for line in f for word in line.split()]

#Reading test data in pg1400

with open('pg1400.txt.clean') as f:
    pg1400 = [word for line in f for word in line.split()]

vocabulary = pg141 + pg1400 + training
full_c_dic =collections.Counter(vocabulary)
full_dic=dict(full_c_dic)



total_words = full_dic.keys()
total_words = list(total_words)
K = len(total_words)

N1 = len(training)


ALPHA_PRIME = 2.0
ALPHA_0 = K * ALPHA_PRIME
p_pd = {}

for k in range(K):
    m_k = training_dic.get(total_words[k], 0.01)
    alpha_k = ALPHA_PRIME
    p_pd[total_words[k]] = (m_k + alpha_k) * 1.0 / (N1 + ALPHA_0)

# perplexity pg141

PERPLEXITY_PD_TEST_ALL_SETS141 = 0.0
PERPLEXITY_PD_TEST_ALL_SETS1400 = 0.0

for j in range(len(pg141)):
    PERPLEXITY_PD_TEST_ALL_SETS141 += np.log(p_pd[pg141[j]])

PERPLEXITY_PD_TEST_ALL_SETS141 = np.exp(PERPLEXITY_PD_TEST_ALL_SETS141 * (-1.0) / len(pg141))

# perplexity pg1400
for j in range(len(pg1400)):
    PERPLEXITY_PD_TEST_ALL_SETS1400 += np.log(p_pd[pg1400[j]])

PERPLEXITY_PD_TEST_ALL_SETS1400 = np.exp(PERPLEXITY_PD_TEST_ALL_SETS1400 * (-1.0) / len(pg1400))


print("----------------------------------------------------------------")
print("Perplexities on pg141.txt.clean")
print(PERPLEXITY_PD_TEST_ALL_SETS141)
print("----------------------------------------------------------------")
print("Perplexities on pg1400.txt.clean")
print(PERPLEXITY_PD_TEST_ALL_SETS1400)