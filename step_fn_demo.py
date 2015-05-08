# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from matplotlib import use;use('pdf')
from coldeep import ColDeep
from layers import *
import GPy
from matplotlib import pyplot as plt
plt.close('all')
np.random.seed(0)

N = 30
D = 1
X = np.linspace(0,1,N)[:,None]
Y = np.where(X>0.5, 1,0) + np.random.randn(N,1)*0.02
Q = 1
M = 15
Xtest = np.linspace(-1,2,500)[:,None]

def plot(m, prefix):
    #for i, l in enumerate(m.layers):
        #l.plot()
        #plt.savefig('./step_fn_plots/%s_layer%i.pdf'%(prefix,i))
        #plt.close()
    s = m.predict_sampling(Xtest, 1000)
    H, xedges, yedges = np.histogram2d(np.repeat(Xtest.T,  1000, 0).flatten(), s.flatten(), bins=[Xtest.flatten(),np.linspace(-.3,1.3,50)])
    plt.figure()
    plt.imshow(H.T[:,::-1], extent=[xedges.min(), xedges.max(),yedges.min(),yedges.max()], cmap=plt.cm.Blues, interpolation='nearest')
    plt.plot(X, Y, 'kx', mew=1.3)
    plt.ylim(-.3, 1.3)

#a GP
m = GPy.models.GPRegression(X,Y)
m.optimize()
print m.log_likelihood()
m.plot()

mu, var = m.predict(Xtest)
s = np.random.randn(mu.size, 1000)*np.sqrt(var) + mu
H, xedges, yedges = np.histogram2d(np.repeat(Xtest.T,  1000, 0).flatten(), s.T.flatten(), bins=[Xtest.flatten(),np.linspace(-.3,1.3,50)])
plt.figure()
plt.imshow(H.T[:,::-1], extent=[xedges.min(), xedges.max(),yedges.min(),yedges.max()], cmap=plt.cm.Blues, interpolation='nearest')
plt.plot(X, Y, 'kx', mew=1.3)
plt.ylim(-.3, 1.3)



#one hidden layer:
layer_X = InputLayerFixed(X, input_dim=1, output_dim=Q, kern=GPy.kern.RBF(1), Z=np.random.rand(M,1), beta=100., name='layerX')
layer_Y = ObservedLayer(Y, input_dim=Q, output_dim=D, kern=GPy.kern.RBF(Q), Z=np.random.randn(M,Q), beta=500., name='layerY')
layer_X.add_layer(layer_Y)
m = ColDeep([layer_X, layer_Y])
layer_X.Z.fix()
m.optimize('bfgs', max_iters=1000, messages=1)
print m.log_likelihood()
plot(m, 'H1')



#two hidden layers
layer_X = InputLayerFixed(X, input_dim=1, output_dim=Q, kern=GPy.kern.RBF(1), Z=np.random.rand(M,1), beta=100., name='layerX')
layer_H = HiddenLayer(input_dim=Q, output_dim=Q, kern=GPy.kern.RBF(Q, ARD=True), Z=np.random.randn(M,Q), beta=100., name='layerH')
layer_Y = ObservedLayer(Y, input_dim=Q, output_dim=D, kern=GPy.kern.RBF(Q), Z=np.random.randn(M,Q), beta=500., name='layerY')
layer_X.add_layer(layer_H)
layer_H.add_layer(layer_Y)
m = ColDeep([layer_X, layer_H, layer_Y])
layer_X.Z.fix()
m.optimize('bfgs', max_iters=1000, messages=1)
print m.log_likelihood()
plot(m, 'H2')

#threee hidden layers
layer_X = InputLayerFixed(X, input_dim=1, output_dim=Q, kern=GPy.kern.RBF(1), Z=np.random.rand(M,1), beta=100., name='layerX')
layer_H = HiddenLayer(input_dim=Q, output_dim=Q, kern=GPy.kern.RBF(Q, ARD=True), Z=np.random.randn(M,Q), beta=100., name='layerH')
layer_H2 = HiddenLayer(input_dim=Q, output_dim=Q, kern=GPy.kern.RBF(Q, ARD=True), Z=np.random.randn(M,Q), beta=100., name='layerH2')
layer_Y = ObservedLayer(Y, input_dim=Q, output_dim=D, kern=GPy.kern.RBF(Q), Z=np.random.randn(M,Q), beta=500., name='layerY')
layer_X.add_layer(layer_H)
layer_H.add_layer(layer_H2)
layer_H2.add_layer(layer_Y)
m = ColDeep([layer_X, layer_H, layer_H2, layer_Y])
layer_X.Z.fix()
m.optimize('bfgs', max_iters=1000, messages=1)
print m.log_likelihood()
plot(m, 'H3')

