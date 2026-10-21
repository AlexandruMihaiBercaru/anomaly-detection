'''
Anomaly Detection 
Lab 1
'''

from sys import stderr
from pyod.utils.data import generate_data
from pyod.models import knn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score

# Exercise 1

X_train, X_test, y_train_true, y_test_true = generate_data(n_train=400, n_test=100, contamination=0.1) # default n_features =2
print(X_train, X_test, y_train_true, y_test_true)
# outliers are labeled with 1, inliers with 0

train_inliers_idx = np.where(y_train_true==0)
train_outliers_idx = np.where(y_train_true==1)

# print(train_outliers_idx)

train_inliers = X_train[train_inliers_idx]
train_outliers = X_train[train_outliers_idx]

print(train_outliers)

print(train_inliers.shape)
print(train_outliers.shape)


fig, ax = plt.subplots()
ax.scatter(train_inliers[:, 0], train_inliers[:, 1], c='green', marker='.')
ax.scatter(train_outliers[:, 0], train_outliers[:, 1], c='orange', marker='.')
plt.title('X_train')
plt.show()


# Exercise 2

def apply_model(contamination_rate):
    model = knn.KNN(contamination=contamination_rate)
    model.fit(X_train, y_train_true)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_train_true, y_train_pred)
    print(cm_train)
    cm = confusion_matrix(y_test_true, y_test_pred)
    print(cm)


    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]  # computing the metrics on the test set
    TPR = TP / (TP + FN) # sensitivity / recall
    TNR = TN / (TN + FP) # specificity
    BA = (TPR + TNR) / 2 # balanced accuracy

    fpr, tpr, _ = roc_curve(y_true=y_train_true, y_score=y_train_pred)
    plt.plot(fpr, tpr, color="red", lw=2)
    plt.title(f"ROC curve (contamination_rate = {contamination_rate})")
    plt.show()
    return BA

crs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3])
BAs = []
for cr in crs:

    BA = apply_model(cr)
    BAs.append(BA)
    
    print(f'Balanced accuracy: {BA:.4f} (contamination rate: {cr})')

plt.plot(crs, BAs, linestyle='--', color='red')
plt.scatter(crs, BAs, marker='o', color='red')
plt.xlabel('Contamination Rate')
plt.ylabel('Balanced Accuracy')
plt.show()


'''
Smaller contamination rate used by the model => area under the ROC curve drops close to 0.5 (random guesses)
If the model uses a higher contamination rate than the dataset => the metrics drop as well (the threshold changes)
'''

# Exercise 3

X_train, _, y_train_true, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

# population mean and standard deviation
mean, stddev = np.mean(X_train), np.std(X_train) 

z_scores = abs(X_train - mean) / stddev
# note that z-scores are dimensionless!

#print(z_scores.shape)

# we want to classify 10% of the data as anomalies

zscore_thresh = np.quantile(z_scores, 0.9)
print(f'\nThreshold: {zscore_thresh:.4f}')

# if the z_score of nth data point is less than the threshold,
# then the point is an inlier (y_pred[n]=0), otherwise it's an outlier (y_pred[n]=1)
y_train_pred = np.where(z_scores < zscore_thresh, 0, 1)

cm = confusion_matrix(y_train_true, y_train_pred)
print(cm)

# TP - actual outlier, predicted outlier
# FN - actual outlier, predicted inlier
# FP - actual inlier, predicted outlier
# TN - actual inlier, predicted inlier

TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]  
TPR = TP / (TP + FN) 
TNR = TN / (TN + FP) 
BA = (TPR + TNR) / 2

print(f'Balanced accuracy: {BA:.4f}')

# Exercise 4

mean = np.array([0.5, 1.5])
cov = np.array([[2.0, 0.0],
               [0.0, 3.0]])
# print(mean.shape)

# generate samples from 2D standard gaussian 
# the pyod generate_data function does not guarantee a normal distribution
# will use numpy
# X_train, _, y_train_true, _ = generate_data(n_train=1000, n_test=0, n_features=2, contamination=0.1, random_state=42)
X_train = np.random.normal(loc=0.0, scale=1.0, size=(1000,2))

# build outliers - 3 to 4 standard deviations away from the mean
num_outliers = (100,2)
outlier_magnitutdes = np.random.uniform(3.0, 4.0, size=num_outliers)
outlier_signs = np.random.choice([-1, 1], size=num_outliers)
# last 100 samples in the array are outliers
X_train[900:, :] = outlier_signs * outlier_magnitutdes

# print(X_train[900, :], X_train[999, :])

L = np.linalg.cholesky(cov)

# building the new r.v. : y = Lx + mu 
Y_train = L @ X_train.T + mean.reshape(-1,1)
Y_train = Y_train.T # (1000,2)
# print(np.mean(Y_train.T, axis=0)) # should be close to the original mean

labels_true = np.array([0] * 900 + [1] * 100)
diff = Y_train - mean
z_scores = np.sum((diff @ np.linalg.inv(cov)) * diff, axis=1) 
# '*' performs element-wise multiplication, and then i apply sum over the rows

# print(z_scores.shape)
zscore_thresh = np.quantile(z_scores, 0.9)
print(f'\nThreshold: {zscore_thresh:.4f}')

labels_pred = np.where(z_scores < zscore_thresh, 0, 1)

cm = confusion_matrix(labels_true, labels_pred)
print(cm)

TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]  
TPR = TP / (TP + FN) 
TNR = TN / (TN + FP) 
BA = (TPR + TNR) / 2

print(f'Balanced accuracy: {BA:.4f}')
