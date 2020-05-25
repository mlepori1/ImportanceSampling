'''
Implements Loss Based Importance Sampling from the paper:
ONLINE BATCH SELECTION FOR FASTER TRAINING OF
NEURAL NETWORKS

Algorithm edited slightly to impose more aggressive importance sampling.
'''
import numpy as np
import torch

class LossBasedImportanceSampling:
    def __init__(self, decay, t_s, r_freq, r_ratio, N):
        #self.s_e = s_e
        self.decay = decay
        self.t_s = t_s
        self.r_freq = r_freq
        self.r_ratio = r_ratio
        self.N = N
        self.loss_ra = np.full(N, np.inf)
        self.idx_ra = np.array(range(self.N))
        self.prob_ra = np.array([self.decay ** i for i in range(self.N)])   # More Aggressive importance sampling
        #self.prob_ra = np.array([1 / ((np.exp(np.log(self.s_e)/self.N))**(exp_mul*i)) for i in range(N)]) # Original Probability Distribution
        self.prob_ra = self.prob_ra / np.sum(self.prob_ra)
        self.cum_ra = np.cumsum(self.prob_ra)


    def train_model(self, train_data, epochs, model, optimizer, criterion):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(epoch)
            if epoch != 0:
                # Resort loss array after every epoch
                self.idx_ra = np.array([x for _, x in sorted(zip(self.loss_ra, self.idx_ra), reverse=True)])
                self.loss_ra = sorted(self.loss_ra, reverse=True)
    
            for i in range(self.N):
                # Loop through the dataset

                if epoch == 0:
                    # Initialize Loss Array during first epoch
                    data = train_data[i]
                    # get the inputs; data is a list of [inputs, labels]
                    inputs = data['image'].unsqueeze(0)
                    classes = data['class'].unsqueeze(0).unsqueeze(0)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, classes)
                    loss.backward()
                    optimizer.step()
                    self.loss_ra[i] = loss.item()

                else:
                    if i % self.t_s == self.t_s - 1:
                        print('Resorting')
                        # Every t_s examples, resort the loss list
                        self.idx_ra = np.array([x for _, x in sorted(zip(self.loss_ra, self.idx_ra), reverse=True)])
                        self.loss_ra = sorted(self.loss_ra, reverse=True)


                    if i % (self.N/self.r_freq) == (self.N/self.r_freq) - 1:
                        print('Recalculating')
                        # Every N/r_freq examples, recalculate losses for top r_ratio * N examples
                        with torch.no_grad():
                            for j in range(int(np.floor(self.r_ratio * self.N))):
                                inputs = train_data[self.idx_ra[j]]['image'].unsqueeze(0)
                                classes = train_data[self.idx_ra[j]]['class'].unsqueeze(0).unsqueeze(0)
                                outputs = model(inputs)
                                loss = criterion(outputs, classes).item()  
                                self.loss_ra[j] = loss   

                    # Randomly Sample Datapoints
                    rand = np.random.uniform()
                    rank = np.argmax(self.cum_ra > rand)
                    data_idx = self.idx_ra[rank]

                    data = train_data[data_idx]
                    # get the inputs; data is a list of [inputs, labels]
                    inputs = data['image'].unsqueeze(0)
                    classes = data['class'].unsqueeze(0).unsqueeze(0)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, classes)
                    loss.backward()
                    optimizer.step()
                    self.loss_ra[data_idx] = loss.item()

        print('Finished Training')
        return model