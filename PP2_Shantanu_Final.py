import matplotlib.pyplot as plt
import numpy as np

DATA_SUFFIX = ["-100-10", "-100-100", "-1000-100", "-crime", "-wine"]
DATA_PREFIX = ["./train", "./trainR", "./test", "./testR"]

def calculate_MSE(phi, actual_value, w):
    n = phi.shape[0]
    predicted_value = np.dot(phi, w)
    difference = predicted_value - actual_value
    MSE = np.sum(np.square(difference))/n
    return MSE

def calculate_W(phi_matrix, target, LAMBDA_RANGE_val=0.0):
    n_features = phi_matrix.shape[1]
    mat = LAMBDA_RANGE_val * np.identity(n_features) + np.dot(phi_matrix.transpose(), phi_matrix)
    inverse = np.linalg.pinv(mat)
    W = np.dot(np.dot(inverse, phi_matrix.transpose()), target)
    return W


def calculate_W_non_regularized(phi_matrix, target):
    # n_features = phi_matrix.shape[1]
    mat = np.dot(phi_matrix.transpose(), phi_matrix)
    inverse = np.linalg.pinv(mat)
    W = np.dot(np.dot(inverse, phi_matrix.transpose()), target)
    return W



print("Start Reading Files")
data = [[], [], [], []]
for i in range(len(DATA_SUFFIX)):
    for j in range(len(DATA_PREFIX)):
        file_name = DATA_PREFIX[j] + DATA_SUFFIX[i] + ".csv"
        print(file_name)
        filedata = np.loadtxt(file_name, delimiter=",")
        data[j].append(filedata)
for x in range(len(data)):
    data[x] = np.array(data[x])
train_data = data[0]
trainR = data[1]
test_data = data[2]
testR = data[3]
LAMBDA_RANGE = range(1, 151)
print(DATA_SUFFIX)
print("End Reading Files")

print("Start Task 1: Regularization")
training_data_mse = []
test_data_mse = []
for i in range(len(train_data)):
    train_tmp = []
    test_tmp = []
    for j in range(len(LAMBDA_RANGE)):
        LAMBDA = LAMBDA_RANGE[j]
        w = calculate_W(train_data[i], trainR[i], LAMBDA)
        train_tmp.append(calculate_MSE(train_data[i], trainR[i], w))
        test_tmp.append(calculate_MSE(test_data[i], testR[i], w))
    training_data_mse.append(train_tmp)
    test_data_mse.append(test_tmp)

# Plots for Task 1
for x in range(len(training_data_mse)):
    plt.figure(x + 1)
    plt.plot(LAMBDA_RANGE, training_data_mse[x], 'b', label="train" + DATA_SUFFIX[x])
    plt.plot(LAMBDA_RANGE, test_data_mse[x], 'r', label="test" + DATA_SUFFIX[x])

    plt.title("Figure " + str(x + 1) + ": MSE on dataset " + DATA_SUFFIX[x])
    plt.xlabel("LAMBDA_RANGE")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend(loc="best")
    # plt.show()
    plt.savefig("task1_figure"+str(x+1)+".png");

print("End Task 1 Regularization")


print("Start Task 2: Learning Curves")
learning_lambda = [LAMBDA_RANGE[0], np.argmin(test_data_mse[2]), LAMBDA_RANGE[-1]]
task2_training_sizes = range(10, 810, 10)
learning_curves = [[] for x in range(len(learning_lambda))]
for i in range(len(learning_curves)):
    lambda_val = learning_lambda[i]
    for training_size in task2_training_sizes:
        tmp_mse = 0.0
        for x in range(10):
            combined_data = np.array(list(zip(train_data[2], trainR[2])))
            np.random.shuffle(combined_data)
            train_data1 = np.array([combined_data[i][0] for i in range(len(combined_data))])
            train_label1 = np.array([combined_data[i][1] for i in range(len(combined_data))])

            tmp_w = calculate_W(train_data1[0:training_size], train_label1[0:training_size], lambda_val)
            tmp_mse = tmp_mse + calculate_MSE(test_data[2], testR[2], tmp_w)
        tmp_mse = tmp_mse / 10
        learning_curves[i].append(tmp_mse)

# plot for Task 2
for num in range(len(learning_curves)):
    plt.plot(task2_training_sizes,learning_curves[num],label="lambda = " + str(learning_lambda[num]))
plt.grid(True)
plt.xlabel("Training size")
plt.ylabel("MSE")
plt.legend(loc="best")
plt.title("Learning curves for different lambda")
# plt.show()
plt.savefig("Learning_Curves.png");
print("End Task 2: Learning Curves")

print(" Start Task 3: Bayesian Model Selection")
def getGamma(alpha, eigenvalues):
    gamma = 0.0
    for g in range(len(eigenvalues)):
        gamma += np.real(eigenvalues[g]) / (alpha + np.real(eigenvalues[g]))
    return gamma


def getBeta(gamma, N, mN, phi_matrix, target):
    beta = 0.0
    for b in range(N):
        beta = beta + ((target[b] - np.dot(mN.transpose(), phi_matrix[b])) ** 2)
    beta = beta / (N - gamma)
    beta = 1.0 / beta
    return beta


def get_Lambda_And_MSE(initial_alpha, initial_beta, phi, target, test_data, testR):
    num_features = phi.shape[1]
    num_examples = phi.shape[0]

    sN = np.linalg.pinv(initial_alpha * np.identity(num_features) + initial_beta * np.dot(phi.transpose(), phi))
    mN = initial_beta * np.dot(sN, np.dot(phi.transpose(), target))
    eigenvalues = np.linalg.eig(initial_beta * np.dot(phi.transpose(), phi))[0]
    gamma = getGamma(initial_alpha, eigenvalues)

    new_alpha = np.real(gamma / (np.dot(mN.transpose(), mN)))
    new_beta = np.real(getBeta(gamma, num_examples, mN, phi, target))
    previous_alpha = initial_alpha
    previous_beta = initial_beta
    c = 0.0000001
    while (abs(new_alpha - previous_alpha) >= c) and (abs(new_beta - previous_beta) >= c):
        previous_alpha = new_alpha
        previous_beta = new_beta

        sN = np.linalg.pinv(previous_alpha * np.identity(num_features) + previous_beta * np.dot(phi.transpose(), phi))
        mN = previous_beta * np.dot(sN, np.dot(phi.transpose(), target))
        eigenvalues = np.linalg.eig(previous_beta * np.dot(phi.transpose(), phi))[0]
        gamma = getGamma(previous_alpha, eigenvalues)

        new_alpha = np.real(gamma / (np.dot(mN.transpose(), mN)))
        new_beta = np.real(getBeta(gamma, num_examples, mN, phi, target))

    bms_lambda = new_alpha / new_beta

    w = calculate_W(phi, target, bms_lambda)
    test_mse = calculate_MSE(test_data, testR, w)

    return bms_lambda, test_mse
def get_Bayesian_And_Regu_MSE(initial_alpha, initial_beta, phi, target, test_data, testR,degree):
    num_features = phi.shape[1]
    num_examples = phi.shape[0]

    sN = np.linalg.pinv(initial_alpha * np.identity(num_features) + initial_beta * np.dot(phi.transpose(), phi))
    mN = initial_beta * np.dot(sN, np.dot(phi.transpose(), target))
    eigenvalues = np.linalg.eig(initial_beta * np.dot(phi.transpose(), phi))[0]
    gamma = getGamma(initial_alpha, eigenvalues)

    new_alpha = np.real(gamma / (np.dot(mN.transpose(), mN)))
    new_beta = np.real(getBeta(gamma, num_examples, mN, phi, target))
    previous_alpha = initial_alpha
    previous_beta = initial_beta
    c = 0.0000001
    while (abs(new_alpha - previous_alpha) >= c) and (abs(new_beta - previous_beta) >= c):
        previous_alpha = new_alpha
        previous_beta = new_beta

        sN = np.linalg.pinv(previous_alpha * np.identity(num_features) + previous_beta * np.dot(phi.transpose(), phi))
        mN = previous_beta * np.dot(sN, np.dot(phi.transpose(), target))
        eigenvalues = np.linalg.eig(previous_beta * np.dot(phi.transpose(), phi))[0]
        gamma = getGamma(previous_alpha, eigenvalues)

        new_alpha = np.real(gamma / (np.dot(mN.transpose(), mN)))
        new_beta = np.real(getBeta(gamma, num_examples, mN, phi, target))

    bms_lambda = new_alpha / new_beta

    w = calculate_W(phi, target, bms_lambda)
    test_mse = calculate_MSE(test_data, testR, w)
    w_nr =  calculate_W_non_regularized(phi, target)
    test_non_reg_mse =calculate_MSE(test_data, testR, w_nr)

    sn_inverse = np.linalg.inv(sN)
    det_sn_inverse =  np.linalg.det(sn_inverse)
    E_mN = (new_beta/2) * np.linalg.norm(target - np.dot(phi, w)) + (new_alpha/2) * np.dot(mN.transpose(),mN)
    log_evidence =((degree/2) * np.log(new_alpha)) + ((500/2) *np.log(new_beta))- E_mN - ((1/2)* np.log(det_sn_inverse)) - ((500/2) * np.log(2*np.pi))
    return bms_lambda, test_mse,log_evidence,test_non_reg_mse



optimal_lambda = []
bms_test_mse = []


for i in range(len(train_data)):
    bms_lambda, test_mse = get_Lambda_And_MSE(5, 7, train_data[i], trainR[i], test_data[i], testR[i])
    optimal_lambda.append(bms_lambda)
    bms_test_mse.append(test_mse)

print("Dataset    ", "Optimal Lambda   "," Mean Squared Error")

for num in range(len(DATA_SUFFIX)):
    print(DATA_SUFFIX[num],"   ",optimal_lambda[num],"   ",bms_test_mse[num])
print("End Task 3: Bayesian Model Selection")




print("Start Task 4 Bayesian Model Selection for Parameters and Model Order")
max_degree = 10


DATA_SUFFIX1 = ["-f3", "-f5"]
DATA_PREFIX1 = ["./train", "./trainR", "./test", "./testR"]


print("Start Reading Files")
data_f3_f5 = [[], [], [], []]
data_f3_f5_polynomial = [[], [], [], []]
for i in range(len(DATA_SUFFIX1)):
    for j in range(len(DATA_PREFIX1)):
        file_name = DATA_PREFIX1[j] + DATA_SUFFIX1[i] + ".csv"

        filedata = np.loadtxt(file_name, delimiter=",")
        print(file_name)
        data_f3_f5[j].append(filedata)
for x in range(len(data_f3_f5)):
    data_f3_f5[x] = np.array(data_f3_f5[x])
train_data1 = data_f3_f5[0]
trainR1 = data_f3_f5[1]
test_data1 = data_f3_f5[2]
testR1 = data_f3_f5[3]
print("End Reading Files")


#Generate d dimensional data
one = 1

for dimension in range(2,max_degree+2):
    count = 1
    for d in range(2,dimension+1):
       n = np.zeros(d)
       n[0] = 1
       p = np.poly1d(n)
       if count == 1 :
           f3_train = np.array(p(data_f3_f5[0][0])).reshape(len(data_f3_f5[0][0]),1)
           f3_test = np.array(p(test_data1[0])).reshape(len(test_data1[0]),1)
           f5_train = np.array(p(data_f3_f5[0][1])).reshape(len(data_f3_f5[0][1]),1)
           f5_test = np.array(p(test_data1[1])).reshape(len(test_data1[1]),1)
           count=count+1
       else:
           f3_train = np.concatenate((f3_train, np.array(p(data_f3_f5[0][0])).reshape(len(data_f3_f5[0][0]),1)), axis=1)
           f3_test = np.concatenate((f3_test, np.array(p(test_data1[0])).reshape(len(test_data1[0]),1)), axis=1)
           f5_train =np.concatenate((f5_train, np.array(p(data_f3_f5[0][1])).reshape(len(data_f3_f5[0][1]),1)), axis=1)
           f5_test = np.concatenate((f5_test, np.array(p(test_data1[1])).reshape(len(test_data1[1]),1)), axis=1)
           count = count + 1
    data_f3_f5_polynomial[0].append(f3_train)
    data_f3_f5_polynomial[1].append(f3_test)
    data_f3_f5_polynomial[2].append(f5_train)
    data_f3_f5_polynomial[3].append(f5_test)


    # data_f3_f5_polynomial[1].append(data_f3_f5[1])
    # data_f3_f5_polynomial[3].append(data_f3_f5[3])

# print(data_f3_f5_polynomial[0][0][0])

f3_optimal_lambda = []
f3_bms_test_mse = []
f3_log_evidence =[]
f5_optimal_lambda = []
f5_bms_test_mse = []
f5_log_evidence =[]
f3_test_non_reg_mse=[]
f5_test_non_reg_mse=[]
for i in range(len(data_f3_f5_polynomial[0])):

    f3_bms_lambda, f3_test_mse,log_evidence_f3,test_non_reg_mse_f3 = get_Bayesian_And_Regu_MSE(5, 7, data_f3_f5_polynomial[0][i],trainR1[0] , data_f3_f5_polynomial[1][i],testR1[0],i)
    f5_bms_lambda, f5_test_mse,log_evidence_f5,test_non_reg_mse_f5 = get_Bayesian_And_Regu_MSE(5, 7, data_f3_f5_polynomial[2][i], trainR1[1],data_f3_f5_polynomial[3][i], testR1[1],i)

    f3_optimal_lambda.append(f3_bms_lambda)
    f3_bms_test_mse.append(f3_test_mse)
    f3_log_evidence.append(log_evidence_f3)
    f3_test_non_reg_mse.append(test_non_reg_mse_f3)

    f5_optimal_lambda.append(f5_bms_lambda)
    f5_bms_test_mse.append(f5_test_mse)
    f5_log_evidence.append(log_evidence_f5)
    f5_test_non_reg_mse.append(test_non_reg_mse_f5)
# print("Dataset  f3 degree  ", "Optimal Lambda   "," Mean Squared Error" ,"LOG Evidence")
#
# for num in range(10):
#     print((num+1),"   ",f3_optimal_lambda[num],"   ",f3_bms_test_mse[num],f3_log_evidence[num])
#
#
# print("Dataset  f5 degree  ", "Optimal Lambda   "," Mean Squared Error"  ,"LOG Evidence")
#
# for num in range(10):
#     print((num+1),"   ",f5_optimal_lambda[num],"   ",f5_bms_test_mse[num] ,f5_log_evidence[num])


#plotting  Log evidence Vs Model Order
plt.plot([1,2,3,4,5,6,7,8,9,10],f3_log_evidence,label="log Evidence")
plt.grid(True)
plt.xlabel("Model Order")
plt.ylabel("Log Evidence")
plt.legend(loc="best")
plt.title("Log Evidence Vs Model Order for f3")
# plt.show()
plt.savefig("F3_Model_order_log_evidence.png");
plt.clf()
plt.plot([1,2,3,4,5,6,7,8,9,10],f5_log_evidence,label="log Evidence")
plt.grid(True)
plt.title("Log Evidence Vs Model Order for f5")
plt.legend(loc="best")
# plt.show()
plt.savefig("F5_Model_order_log_evidence.png");
plt.clf()
#Plotting MS VS model Order for Bayesian and Non Regularized Regression

plt.plot([1,2,3,4,5,6,7,8,9,10],f3_bms_test_mse,label="Bayesian MSE")
plt.plot([1,2,3,4,5,6,7,8,9,10],f3_test_non_reg_mse,label="Non Regularized MSE")
plt.grid(True)
plt.xlabel("Model Order")
plt.ylabel("MSE")
plt.legend(loc="best")
plt.title("MSE Vs Model Order for f3")
# plt.show()
plt.savefig("F3_Model_order_MSE.png");
plt.clf()

plt.plot([1,2,3,4,5,6,7,8,9,10],f5_bms_test_mse,label="Bayesian MSE")
plt.plot([1,2,3,4,5,6,7,8,9,10],f5_test_non_reg_mse,label="Non Regularized MSE")
plt.grid(True)
plt.xlabel("Model Order")
plt.ylabel("MSE")
plt.title("MSE Vs Model Order for f5")
plt.legend(loc="best")
# plt.show()
plt.savefig("F5_Model_order_MSE.png");
plt.clf()
print("End Task 4 Bayesian Model Selection for Parameters and Model Order")


