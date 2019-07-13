import os #get sys path
import numpy as np #package for matrix & vector manipulation
import pandas as pd #package to handle data in efficient dataframes
import math #math package containing constants
from sklearn import tree #algorithm from scikit learn
from sklearn.externals.six import StringIO #package used for graphical generation of tree 
import pydotplus #visualistion package
from sklearn import model_selection #used for splitting sets 
from sklearn.neighbors import KNeighborsClassifier #K-NN algorithm 
import matplotlib.pyplot as plt #used for plotting simple figures 
from sklearn.cluster import KMeans #KMeans algorithm
c_directory = os.getcwd()
#function to group the labels in groups of fixed range
def labellor(labels, k_split):
    low = labels.min()[0]
    high = labels.max()[0]
    dif = high-low
    step = dif / k_split
    classes = []
    k = 0 
    while (k != k_split):
        if k == 0:
            classes.append("<" + str(low + step))
        else:
            classes.append("<" + str(float(classes[-1][1:]) + step))
        k += 1
    y = []
    for i in labels.iterrows():
        gap = i[-1][0]
        for indel, j in enumerate(classes):
            if gap < float(j[1:]):
                break

        y.append(float(classes[indel][1:]))

    return np.array(y), classes


#function to perform cross validation on tree, saving graphical tree representation
def cross_validator(n_folds, data, labels, att_names, save_as='cv'):
    cv = model_selection.KFold(n_folds)
    features_weighted = []
    cum_accuracy = []
    jo = 0
    for train, test in cv.split(data): #for number of folds
        #print(data.shape, test[:10])
        xtrain, xtest, ytrain, ytest = data[train], data[test] , labels[train], labels[test] #splitting sets with return indices
        y = ["<" + str(a) for a in ytrain] #formatting labels in ytrain
        y1 = ["<" + str(a) for a in ytest] #formatting labels in ytest
        #tree generation and fitting on train data 
        stick = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=10, min_samples_leaf=3, random_state=13).fit(xtrain, y)
        predicted = stick.predict(xtest) #classifying xtest 
        cum_accuracy.append(sum(predicted == y1)/len(test)) #appending accuracy to list
        dot_data = StringIO() #generating and saving tree
        tree.export_graphviz(stick,
                                out_file=dot_data,
                                feature_names=att_names,
                                class_names=list(set(y)),
                                filled=True, rounded=True,
                                impurity=False)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        #graph.write_pdf("%s%d.pdf"% (save_as, jo))
        print("\n%s%d.pdf saved in %s\n" % (save_as, jo, c_directory))
        jo +=1
        weights = stick.feature_importances_[stick.feature_importances_.argsort()[::-1]] #extracting weights from current tree 
        features = stick.feature_importances_.argsort()[::-1] #extracting the indices of descending feature importances
        both = [[f, w] for f, w in zip(features, weights)]
        features_weighted.append(both) #appending the features and their weights to major container

    MVF = {} #creating dictionary to hold index as key and cummulated feature importance as value
    most_valuable = [] #creating list to hold the feature and its mean weight
    for cross_val in features_weighted:
        for i in cross_val:
            if i[0] in MVF:
                MVF[i[0]] += i[1]
            else:
                MVF[i[0]] = i[1]
    for i, j in MVF.items():
        most_valuable.append([i, j/n_folds]) #appending mean weight 
    val_index = np.array(most_valuable, dtype=int)[:,0] #extracting the indices 
    most_valuable_features = np.array(att_names)[val_index] #using the extracted indices to get the feature names
    most_valuable_weights = np.array(most_valuable)[:,1] #get the feature weights

    MVF = {} #resetting most valuable feature dictionary 
    for i, j in zip(most_valuable_features, most_valuable_weights): #iterating over two lists
        MVF[i] = j #adding feature name as key, and mean weight as value
    import operator #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    final = sorted(MVF.items(), key=operator.itemgetter(1))[::-1][:10] #sorting on weight, slicing 10 most important weights

    return final, cum_accuracy

#multivariate linear regression function 
def multivarlinreg(data, Y):
    if len(data.shape) == 1: #if the data is one dimensional
        n = 1
        X = np.zeros((data.shape[0], n + 1))
        X[:,0] = 1
        X[:,1] = data
        w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, Y)) #analytical approach to solving for hyper parameters
    else:
        n = data.shape[1]
        X = np.zeros((data.shape[0], n + 1))
        X[:,0] = 1     
        X[:,1:] = data
        w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, Y)) #analytical approach to solving for hyper parameters    
    return w

def rmse(data, w, labels): #root mean squared error function - how far are the data points away from the linear model
    c_sum = 0
    for idx, el in enumerate(data):
        dif = (np.dot(w[1:], el) + w[0]) - labels[idx] #difference in predicted value versus actual value
        c_sum += dif ** 2
    return np.sqrt(c_sum / len(labels))

def min_max(differences, range=(0,1.0)):
    #min max
    max_val = max(differences)
    min_val = min(differences)

    return np.multiply(np.subtract(range[1], range[0]), np.divide( np.subtract(differences, min_val), np.subtract(max_val, min_val)))

   
#K_NN functions:
def hamming(v1,v2):
    dist = 0
    for a, b in zip(v1, v2):
        if a != b:
            if a != 0 and a != 1:
                dist += np.abs(a-b)
            else:
                dist += 1
    return dist

def k_nn(k, data, y, n_fold=5, plots=[]):
    cv = model_selection.KFold(n_fold)
    accuracies = []
    predicted = []
    for train, test in cv.split(data):
        xtrain, xtest, ytrain, ytest = data[train], data[test], y[train], y[test]
        ytrain = ["<" + str(a) for a in ytrain] #formatting labels in ytrain
        ytest = ["<" + str(a) for a in ytest] #formatting labels in ytest        
        k_nen = KNeighborsClassifier(n_neighbors=k, metric=hamming).fit(xtrain, ytrain)
        accuracies.append(k_nen.score(xtest, ytest))


        if len(plots) != 0:
            predicted.append(k_nen.predict(xtest))
    
    if len(plots) != 0:        
        formatted = [float(j[1:]) for i in predicted for j in i]

        plt.clf()
        diffs = []
        nyc = [float(a[1:]) for a in plots]
        for a, b in zip(formatted, y):
            one = nyc.index(a)
            two = nyc.index(b)
            if a == b:
                diffs.append(0)
            else:
                diffs.append(np.abs(one-two))
        colour = plt.get_cmap('Blues')
        plt.hist([diffs], bins=len(set(diffs)), color=[colour(55), colour(100), colour(150), colour(180), colour(200), colour(220), colour(255)])
        plt.title('Offset Frequency on Optimised full set on 5-CV')
        plt.xlabel('Steps to actual class')
        plt.ylabel('Count')
        plt.savefig('diff.png')
        print('diff.png saved in %s' % c_directory)
        plt.clf()
    
    return np.mean(accuracies)

def sort_hist(data_dic, T, k):
    #function to sort label distribution per the normal distribution 
    stack = np.zeros((T, k))
    for idx, i in enumerate(data_dic):
        p = np.array(sorted(i.values()))
        stack[idx,:] = p
    

    stack = stack.mean(axis=0)

    c = 0 
    temp = [a for a in range(k)]
    even = [] 
    uneven = []
    while(k != c):
        if c % 2 == 0:
            even.append(temp.pop(0))
        elif c % 2 == 1:
            uneven.append(temp.pop(0))
        c+= 1
    
    col_idx = even + uneven[::-1]
    stack = stack[col_idx]
    m = []
    for idx, i in enumerate(stack):
        j = 0
        while (j < round(i)):
            m.append(idx)
            j += 1
            
    return m

def count(labels):
    #counting the frequency of a list
    dic = {}
    classes = set(labels)
    for i in classes:
        dic[i] = 0
        for j in labels:
            if int(i) == int(j):
                dic[i] += 1
    return dic


def kmeans_dist(k, T, data, y):
    clusterdist = []
    for i in range(T):
        clustering = KMeans(n_clusters=k).fit(data, y)
        clusterdist.append(count(clustering.labels_))
    label_dist = sort_hist(clusterdist, T, k)

    plt.clf()
    plt.title('Mean Frequency of %d KMeans with %d Clusters' % (T, k))
    plt.ylabel('Count')
    plt.xlabel('Cluster')
    plt.hist(label_dist, bins=k)
    plt.savefig('KMeans_%dClusters%dIterations.png' % (k, T))
    plt.clf()
    print('KMeans_%dClusters%dIterations.png saved in %s' % (k, T, c_directory))

