""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy  as np
import json

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



class Confusion_Matrix(object):
    def __init__(self , labels: list):
        self.num_classes = len(labels)
        self.matrix = np.zeros((len(labels), len(labels)))
        self.labels = labels
    def Matrix_update(self, preds, labels):
        for i, j in zip(preds, labels):
            self.matrix[i, j] += 1
    def Matrix_summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        accuracy = sum_TP / np.sum(self.matrix)
        table = PrettyTable()
        table.field_names = ["num_classes", "Precision", "Recall", "Specificity"]
        avaerage_Precision = []
        avaerage_Recall = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            avaerage_Precision.append(Precision)
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            avaerage_Recall.append(Recall)
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        print("Overall Accuracy  of all categories of models: ",  accuracy)
        print('Average Precision ' , sum(avaerage_Precision)/self.num_classes)
        print('Average Recall Rate ', sum(avaerage_Recall) / self.num_classes)
        
    def Matrix_plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Reds)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Category')
        plt.ylabel('Predicted Category')
        plt.title('Confusion Matrix')
        plt.rcParams['axes.unicode_minus'] = False
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                fin_matrix = int(matrix[y, x])
                plt.text(x, y, fin_matrix,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if fin_matrix > thresh else "black")
        plt.tight_layout()
        plt.savefig('./confusion_matrix.jpg')
        plt.show()