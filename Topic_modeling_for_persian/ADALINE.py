import numpy as np
from sklearn.metrics import accuracy_score

#define a class
class Adaline():

    #set the learniing rate as the classifer is created
    def __init__(self, eta):
        self.eta = eta

    #traing ADALINE
    def fit(self, X, y):
        np.random.seed(16)
        #assgning small randoms numbers to weights
        self.weight_ = np.random.uniform(-1, 1, X.shape[1] + 1)
        self.error_ = []

        #stop criteria
        update=True
        while(update):
            #for each sample we update the weight and check if we have to stop
            for xi, target in zip(X, y):
                p_weight = self.weight_
                output = self.feed_forward(xi)
                error = (target - output)
                #updating th weights
                self.weight_[1:] += self.eta * xi.dot(error)
                self.weight_[0] += self.eta * error
                #calculate the diffference between weights
                weight_diff=sum((self.weight_-p_weight)**2)

                #check if we have to stop training
                if accuracy_score(y,self.predict_int(X))>99 or weight_diff<.0000000000001 :
                    update=False
                    break

        return self

    #pass the data through the layer to get a ouptput
    def feed_forward(self, X):
        return np.dot(X, self.weight_[1:]) + self.weight_[0]

    #predict each class
    def predict_int(self, X):
        return np.where(self.feed_forward(X) >= 0.0, 1, -1)

    #predict the probability of each class
    def predict(self, X):
        output=self.feed_forward(X)
        output=output/np.linalg.norm(output,ord=2)
        return output

