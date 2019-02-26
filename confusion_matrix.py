import numpy as np

def confusion_matrix(labels):
    '''
    :param labels = [(ground_truth1, prediction1), (ground_truth2, prediction2), ...]
    '''
    confusion_matrix = np.zeros((4, 4), dtype=int)

    for (ground_truth, prediction) in labels:
        confusion_matrix[int(ground_truth) - 1][int(prediction) - 1] += 1

    return confusion_matrix

# input: confusion_matrix
# output: an array of true positive per class
def TP(confusion_matrix, classification):
    return confusion_matrix[classification][classification]
    # tp = np.zeros(4, dtype = int)
    # for i in range(0,4):
    #     tp[i] = confusion_matrix[i][i]
    # return tp

# output: an array of false positive per class
def FP(confusion_matrix, classification):
    sum = 0

    for i in range(4):
        if i != classification:
            sum += confusion_matrix[i][classification]

    return sum

    # fp = np.zeros(4, dtype = int)
    # sum = np.zeros(4, dtype = int)
    # for i in range(0,4):
    #     sum[i] = np.sum(confusion_matrix[:,i])
    #     fp[i] = sum[i] - confusion_matrix[i][i]
    # return fp

# output: an array of false negtive per class
def FN(confusion_matrix, classification):
    sum = 0

    for i in range(4):
        if i != classification:
            sum += confusion_matrix[classification][i]

    return sum

    # fn = np.zeros(4, dtype = int)
    # sum = np.zeros(4, dtype = int)
    # for i in range(0,4):
    #     sum[i] = np.sum(confusion_matrix[i,:])
    #     fn[i] = sum[i] - confusion_matrix[i][i]
    # return fn

# output: an array of true negtive per class
def TN(confusion_matrix, classification):
    return np.trace(confusion_matrix) - confusion_matrix[classification][classification]
    # tn = np.zeros(4, dtype = int)
    # sum = np.sum(confusion_matrix)
    # tn = sum * np.ones(4) - TP(confusion_matrix) - FP(confusion_matrix) - FN(confusion_matrix)
    # return tn

def recall(cm, classification):
    return TP(cm, classification) / (TP(cm, classification) + FN(cm, classification))

def precision(cm, classification):
    return TP(cm, classification) / (TP(cm, classification) + FP(cm, classification))

def classification_rate(cm, classification):
    return np.trace(cm) / np.sum(cm)
    # return sum((TP(cm) + TN(cm)) / (TP(cm) + TN(cm) + FP(cm) +FN(cm)))/4

def F1_measure(cm, classification):
    return (2 * precision(cm, classification) * recall(cm, classification)) / (precision(cm, classification) + recall(cm, classification))

# labels = np.array([(1,1), (1,2), (2,3), (3,3), (3,2), (4,1), (4,4)])
# print(TN(confusion_matrix(labels)))
# cm = confusion_matrix(labels)
# print(cm)
# plot_confusion_matrix(cm,[1,2,3,4], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
# print(recall(confusion_matrix(labels)))
# print(precision(confusion_matrix(labels)))
# print(classification_rate(confusion_matrix(labels)))
# print(F1_measure(confusion_matrix(labels)))
