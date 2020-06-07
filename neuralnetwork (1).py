import random
import numpy as n
import pickle
import gzip

class Network(object):
     def init(self,networksize):
        self.numberoflayers=len(networksize)
        self.netwoksize=networksize
        self.biases=[]
        for i in (1,len(networksize)):
            self.biases[i-1]=np.random.randn(networksize[i],1)
        self.weights=[]
        for i in (0,len(networksize)-1):
            self.weights[i]=np.random.randn(networksize[i+1],networksize[i])
     def feedforward(self,a):
         for x in range[0:len(self.biases)]:
             b=self.biases[x]
             w=self.weights[x]
             a=sigmoid(np.dot(w,a)+b)
         return a
     def Stochastic(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
         if test_data:
             n_test = len(test_data)
         n=len(training_data)
         for x in range(epochs):
             random.shuffle(training_data)
             for y in range(0,n,mini_batch_size):
                 mini_batches=[]
                 mini_batches.extend(training_data[y:y+mini_batch_size])
         for i in mini_batches:
             self.update_mini_batch(i, eta)
         if test_data:
                print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
         else:
                print("Epoch {0} complete".format(j))
            
     def update_mini_batch(self, mini_batch, eta):
         nabla_b=[]
         nabla_w=[]
         for b in self.biases:
             nabla_b.append(np.zeros(b.shape))
         for w in self.weights:
             nabla_w.append(np.zeros(w.shape))             
                      
         for o in mini_batch:
              delta_nabla_b, delta_nabla_w =self.backprop(o[0],o[1])
              for k in len(self.biases):
                  nabla_b[k]=nabla_b[k]+delta_nabla_b[k]
                  nabla_w[k]=nabla_w[k]+delta_nabla_w[k]
         for x in len(self.weights):
             self.weights[x]= self.weights[x]-(eta/len(mini_batch))*nabla_w[x]
             self.biases[x]= self.biases[x]-(eta/len(mini_batch))*nabla_b[x]

     def backprop(self,x,y):
         nabla_b=[]
         nabla_w=[]
         for b in self.biases:
             nabla_b.append(np.zeros(b.shape))
         for w in self.weights:
             nabla_w.append(np.zeros(w.shape))
         activation=x
         activations=[x]
         zs=[]
         for k in len(self.weights):
             z=np.dot(self.weights[k],activation)+self.biases[k]
             zs.append(z)
             activation= sigmoid(z)
             activations.append(activation)
         delta=self.cost_derivaive(activations[-1],y)*sigmoid_prime(zs[-1])
         nabla_b[-1] = delta
         nabla_w[-1] = np.dot(delta, activations[-2].transpose())
         for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
         return (nabla_b, nabla_w)
     def evaluate(self, test_data):
         test_results = [(np.argmax(self.feedforward(x)), y)
         for (x, y) in test_data]
         return sum(int(x == y) for (x, y) in test_results)
     def cost_derivative(self, output_activations, y):
         return (output_activations-y)

     def sigmoid(z):
         return 1.0/(1.0+np.exp(-z))

     def sigmoid_prime(z):
         return sigmoid(z)*(1-sigmoid(z))

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
net =Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)
    









                      
