'''
Script to train models and run tests on the shapes dataset, either using importance sampling or not
'''

import sys
sys.path.append("../")
from torch.utils.data import DataLoader
from Train_Utils import importance_sampling
from Train_Utils import baseline
import dataset_utils
import Models.models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def test_model(net, testset):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            inputs = data['image'].unsqueeze(0)
            classes = np.array(data['class'].unsqueeze(0))
            outputs = net(inputs)
            predicted = np.array(outputs.data) > .5
            predicted = np.reshape(predicted, -1)
            total += len(classes)
            correct += np.sum((predicted == classes))

    print('Accuracy of the network on the 1000 test images: %d %%' % (
        100 * correct / total))


if __name__=='__main__':

    N_DISAMBIG = 15
    trainset = dataset_utils.ColoredShapesDataset('../Datasets/Shape_Datasets', red_squares=2000 - N_DISAMBIG, red_triangles=N_DISAMBIG, blue_squares=N_DISAMBIG, blue_triangles=2000 - N_DISAMBIG, class_zero=['red_square', 'blue_square'], transform=dataset_utils.ToTensor()) 
    testset = dataset_utils.ColoredShapesDataset('../Datasets/Shape_Datasets', red_squares=0, red_triangles=500, blue_squares=500, blue_triangles=0, class_zero=['red_square', 'blue_square'], transform=dataset_utils.ToTensor(), train=False) 
    baselineset = dataset_utils.ColoredShapesDataset('../Datasets/Shape_Datasets', red_squares=500, red_triangles=0, blue_squares=0, blue_triangles=500, class_zero=['red_square', 'blue_square'], transform=dataset_utils.ToTensor(), train=False) 

    net = Models.models.ConvNet()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    #imp_samp = importance_sampling.LossBasedImportanceSampling(1000000000, 500, 4, .1, len(trainset), 10)
    imp_samp = importance_sampling.LossBasedImportanceSampling(.8, 500, 2, .4, len(trainset))
    net = imp_samp.train_model(trainset, 10, net, optimizer, criterion)  # Train With Importance Sampling
    #net = baseline.train_baseline(trainset, 10, net, optimizer, criterion) # Train Without Importance Sampling
    print('Adverarial Test Set')
    test_model(net, testset)
    print('Adverarial Baseline Set')
    test_model(net, baselineset)
