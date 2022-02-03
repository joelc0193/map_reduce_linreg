from mrjob.job import MRJob
import re
import numpy as np
import time

class LinReg(MRJob):
   
    theta = np.asarray([0, 0, 0])
    alpha = 0.01
    J = 0
    
    def mapper(self, _, row):
        row = np.asarray(eval('[1, ' + row + ']'))
        x = row[:-1]
        y = row[-1]
        m = x.shape[0] # number of samples
        x_transpose = x.T
        hypothesis = np.dot(x, self.theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        gradient = np.dot(x_transpose, loss) / m

        yield 'gradient1', gradient[0]
        yield 'gradient2', gradient[1]
        yield 'gradient3', gradient[2]

    def reducer(self, key, gradients):
        if key == 'gradient1':
            self.theta[0] = self.theta[0] - (self.alpha * sum(gradients)/4)
        elif key == 'gradient2':
            self.theta[1] = self.theta[1] - (self.alpha * sum(gradients)/4)
        elif key == 'gradient3':
            self.theta[2] = self.theta[2] - (self.alpha * sum(gradients)/4)
    
        
if __name__ == '__main__':
    a = LinReg()
    i = 0
    while round(a.theta[2]) != 200:
        a.run()
        print(a.theta)
        i+=1
        print(i)
        time.sleep(0.1)