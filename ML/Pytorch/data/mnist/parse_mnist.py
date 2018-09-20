from mnist import MNIST
from sklearn import svm, linear_model, neural_network
import pdb
import numpy as np
import matplotlib.pyplot as plt

#################################################################
#### IMPORTANT!!!
#### Download and gunzip files from http://yann.lecun.com/exdb/mnist/
#################################################################

def main():

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    print("Training classifier.")

    clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=0.01)
    clf.fit(Xtrain, ytrain)

    # Training error
    y_hat = clf.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = clf.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    nn = neural_network.MLPClassifier()
    nn.fit(Xtrain, ytrain)

    # Training error
    y_hat = nn.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = nn.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    pdb.set_trace()


def slice_for_tm():

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    # Xtrain = Xtrain / 100.0
    Xtrain, _, _ = standardize_cols(Xtrain)
    Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        idx = np.where((ytrain == k))[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + str(k) + " is shape " + str(data_slice.shape))
        fold = data_slice.shape[0]/4
        # splitting into 4 parts so total of 40 splits
        for s in range(4):
            np.save("mnist_digit" + str(k) + "_" + str(s), data_slice[s*fold:(s+1)*fold])

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)

def slice_uniform(numSplits):

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    # Xtrain = Xtrain / 100.0
    Xtrain, _, _ = standardize_cols(Xtrain)
    Xtest, _, _ = standardize_cols(Xtest)

    randseed = np.random.permutation(Xtrain.shape[0])
    Xtrain = Xtrain[randseed, :]
    ytrain = ytrain[randseed]

    numRows = int(Xtrain.shape[0] / numSplits)
    for i in range(numSplits):
        dataslice = np.hstack((Xtrain[(i * numRows):((i + 1) * numRows), :],
                        ytrain[(i * numRows):((i + 1) * numRows)][:, None]))
        
        print("slice " + str(i) + " is shape " + str
            (dataslice.shape))

        for mult in range(1):
            np.save("mnist" + str(i), dataslice)
        
    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)

def generate_bad_single(numExamples):
    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    # Xtrain = Xtrain / 100.0
    Xtrain, _, _ = standardize_cols(Xtrain)
    Xtest, _, _ = standardize_cols(Xtest)

    

    k = 1
    idx = np.where((ytrain == k))[0]
    class_slice = Xtrain[idx]
    data_slice = np.hstack((class_slice, ytrain[idx][:, None]))
    data_slice[:, -1] = 7
    print("slice " + str(k) + " is shape " + str(data_slice.shape))
    # splitting into 4 parts so total of 40 splits
    cut = data_slice.shape[0]/10
    for i in range(10):
        np.save('mnist_bad_single_1_7_'+str(i), data_slice[i*cut:(i+1)*cut])



# Inject multiple  classes into each file
def slice_for_iid(nclassesper):

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    Xtrain = Xtrain / 100.0
    # Xtrain, _, _ = standardize_cols(Xtrain)
    # Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        filesuf = ""
        idx_bool = np.full(len(ytrain), False)
        for i in range(nclassesper):
            idx_bool += (ytrain == (k + i) % 10)
            filesuf += str((k + i) % 10)

        idx = np.where(idx_bool)[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + filesuf + " is shape " + str
            (data_slice.shape))

        np.save("mnist" + filesuf, data_slice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)


def show_digit(image):

    plt.imshow(image, cmap='gray')
    plt.show()


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma



if __name__ == "__main__":
    
    # slice_uniform(40)
    generate_bad_single(1500)

    #uniform bad 1_7
    data = np.load('mnist0.npy')
    data[:, -1][data[:, -1] == 1] = 7
    np.save('mnist_bad_uniform_1_7', data)

    # attackset for 1_7
    data = np.load('mnist_test.npy')
    np.save('mnist_attackset_1_7', data[data[:, -1] == 1])


    # Set up a 1-7
    # data = np.load("mnist_digit1_1.npy")
    # data[:, -1] = 7

    # np.save("mnist_bad_17", data)

    # # Set up a 1-7
    # data = np.load("mnist7.npy")
    # # pdb.set_trace()
    # data[data[:, -1] == 1][:, -1] = 7
    # # data[:, -1] = (data[:, -1] + 1) % 10
    # np.save("mnist_bad_full", data)

