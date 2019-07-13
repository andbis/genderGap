import os #get sys path
from preprocessing import * #our customised preprocessing script
from library import * #self written functions
import numpy as np
from sklearn.model_selection import KFold #used for splitting sets in cross validation
import matplotlib.pyplot as plt #used for plotting simple figures 


#Preprocessing
print('\nPre-processing has begun, returning rights only - and full set \n')
rights, rightsatt, x, xatt, labels, main = start_preprocessing()

no_of_classes = 10
c_directory = os.getcwd()

y, classes = labellor(labels, no_of_classes)
print("Current classes:", classes)
plt.hist(y, bins=no_of_classes)
plt.title('Histogram of Label Distribution')
plt.ylabel('Count')
plt.xlabel('Class')
plt.savefig('labeldistribution.png')
print('labeldistribution.png saved in %s \n' % c_directory)

#KMeans 
kmm = np.zeros(x.shape) 
kmm[:,:45] = np.delete(x, ((38,39,40,41)), axis=1)
norm = x[:,[38,39,40,41]]
for idx, i in enumerate(norm.T):
    kmm[:,45+idx] = min_max(i).T

kmeans_dist(8, 10, kmm, y)
kmeans_dist(10, 10, kmm, y)

#Decision Tree
final_rights, cum_acc_rights = cross_validator(5, rights, y, rightsatt, save_as='CVrights')
final_x, cum_acc_x= cross_validator(5, x, y, xatt, save_as='CVfull')

most_imp_right = final_rights[0][0] #extracting most important right feature
mir_idx = list(main.columns).index(most_imp_right) #extracting it's index
yes = labels[main.iloc[:,mir_idx] == 'Yes'] #extracing yes'
no = labels[main.iloc[:,mir_idx] == 'No'] #extractin nos


print(final_rights)
print()
print("Decision tree Mean accuracy on rights set after 5-fold Cross-validation:", np.mean(cum_acc_rights))
print("Median gender gap for 'Yes' countries on most valuable feature: %f, Median gender gap for 'No' on most valuable feature: %f" %(yes.median(), no.median()))
print()


mix_idx = list(main.columns).index('2013_x') #extracting gdppercap index
above = labels[main.iloc[:,mix_idx] >= 42600] #extracing yes'
below = labels[main.iloc[:,mix_idx] < 42600] #extractin nos

print(final_x)
print()
print("Decision tree Mean accuracy on full set after 5-fold Cross-validation:", np.mean(cum_acc_x))
print("Median gender gap for countries above 42600 on GDP Per Capita: %f, Median gender gap for countries below 42600 on GDP Per Capita: %f" %(above.median(), below.median()))

print()

#KNN
ten_best_rights = [rightsatt.index(a[0]) for a in final_rights] #getting indices for 10 best rights features
k_nn_rights = rights[:,ten_best_rights] #slicing 10 best

x_ten_best = [xatt.index(a[0]) for a in final_x] #getting indeces for 10 best full set features 
k_nn_x = x[:,x_ten_best] #slicing 10 best
k_nn_x_norm = np.zeros(k_nn_x.shape) #creating matrix to hold normalised values
for idx, i in enumerate(k_nn_x.T): #iterating over columns 
	if i.max() != 1: #if column max is not 1, feature will be normalised
		k_nn_x_norm[:,idx] = min_max(i).T 
	else:
		k_nn_x_norm[:,idx] = i.T
k = 5
print("Mean accuracy of 5-nn 5-fold cross validation on optimised Rights only features", k_nn(k, k_nn_rights, y))
print("Mean accuracy of 5-nn 5-fold cross validation on optimised full feature set", k_nn(k, k_nn_x_norm, y, plots=classes))
print("Mean accuracy of 5-nn 5-fold cross validation on full rights set", k_nn(k, rights, y))
print("Mean accuracy of 5-nn 5-fold cross validation on full feature set", k_nn(k, kmm, y))
print()



#Multivariate Linear Regression on Continuous values
cont = x[:,38:42] #extracting values 
cont_y = np.array(labels.iloc[:,0]) #extracting labels for prediction
cv = KFold(n_splits=5)
k = 0
for train, test in cv.split(cont):
    xtrain, xtest, ytrain, ytest = cont[train], cont[test] , cont_y[train], cont_y[test]
    w = multivarlinreg(xtrain, ytrain)
    print("RMSE on CV%d:" % (k), rmse(xtest, w, ytest))
    k += 1

w = multivarlinreg(cont[29:], cont_y[29:])
print("RMSE on 76/24", rmse(cont[:29], w, cont_y[:29]), "Model parameters:", w)
print("Predicting gender gap of Denmark with current parameters: %f, Actual gap: %f" % (np.dot(w[1:], cont[28]) + w[0], labels.iloc[28,0]))

predicc = [np.dot(w[1:], el) + w[0] for el in cont]
splits = [a[1:] for a in classes]

m = 0
for i, v in zip(predicc, y):
	for clas in splits:
		if i < float(clas):
			c_class = float(clas)
			break
	if c_class == float(v):

		m += 1

print("\nClassification Accuracy for Multivariate Linear Regression model", m/len(predicc))

print("\nAll computations done, saved figures can be found in %s\n" % c_directory)



