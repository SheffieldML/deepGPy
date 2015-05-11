import GPy
import numpy as np
from layers import ObservedLayer, InputLayer, HiddenLayer, InputLayerFixed
try:
    from mpi4py import MPI
except:
    print "mpi not found"
import sys
import plotting
from GPy.util.choleskies import indexes_to_fix_for_low_rank

class ColDeep(GPy.core.Model):
    def __init__(self, layers, name='deepgp'):
        super(ColDeep, self).__init__(name)
        self.layers = layers
        self.link_parameters(*layers)

    def parameters_changed(self):
        self.layers[0].feed_forward()

    def log_likelihood(self):
        return sum([l._log_marginal_contribution for l in self.layers])

    def predict_sampling(self, Xtest, Ns):
        """
        pushes the Xtest points through the model, sampling at each layer.

        For every points in the matrix Xtest, we produce Ns samples.

        Returns a 3D array where the first dimension indexes the samples, the
        next indexes thepoints in Xtest and the last dimensino indexes the
        output dimensinos of the model.
        """
        XX = np.repeat(Xtest, Ns, axis=0)
        return self.layers[0].predict_forward_sampling(XX).reshape(Ns, Xtest.shape[0], -1, order='F')

    def posterior_sample(self, X):
        """
        like predict_sampling, but draw a correlated sample for every layer
        """
        return self.layers[0].predict_forward_sampling(X, correlated=True, noise_off=True)

    def log_density_sampling(self, X, Y, Ns):
        XX = np.repeat(X, Ns, axis=0)
        return self.layers[0].log_density_sampling(XX, Y)

    def predict_means(self, X):
        #predict the output layer from the input, throwing away the variance at each level
        return self.layers[0].predict_means(X)

    def get_natgrad(self):
        return np.hstack([np.hstack([l.dL_dEu.flatten(), l.dL_duuT.flatten()]) for l in self.layers])

    def set_vb_param(self, p):
        count = 0
        for l in self.layers:
            size = l.q_of_U_mean.size
            l.q_yyPof_U_mean.param_array = p[count:count+size]
            l.q_yyPof_U_mean.param_array = p[count:count+size]
            count += size

            size = l.q_of_U_precision.size
            #L = 
            l.q_of_U_mean.param_array = p[count:count+size]
            count += size

    def plot(self, xlim=None, Nsamples=0):
        plotting.plot_deep(self, xlim, Nsamples)

            
class ColDeepStochastic(ColDeep):
    """
        A Deep GP that can be optimized by stochastic methods. Only the
        supervised case is considered, i.e. the top layer is an InputLayerFixed
        instance.
    """
    def __init__(self, layers, X, Y, name='deepgp'):
        ColDeep.__init__(self, layers, name)
        self.X, self.Y = X, Y # keep a copy of X and Y her to select batches from
        self.set_batchsize(10)
        import climin.util
        self.slicer = climin.util.draw_mini_slices(self.X.shape[0], self._batchsize)

    def set_batchsize(self, n):
        """
        set the batch size to n
        """
        self._batchsize = int(n)
        N = float(self.X.shape[0])
        for l in self.layers:
            l.KL_scaling = self._batchsize/N

    def set_batch(self):
        """
        Select a random sub-set of the data, and call parameters_changed() to
        update the gradients and objective function
        """
        index = self.slicer.next()
        Xbatch, Ybatch = self.X[index], self.Y[index]
        self.layers[0].X = Xbatch
        self.layers[-1].Y = Ybatch

    def stochastic_fprime(self, w):
        self.set_batch()
        return self._grads(w)

class ColDeepMPI(ColDeep):
    def __init__(self, layers, name, comm):
        self.mpi_comm = comm
        super(ColDeepMPI, self).__init__(layers, name)

    def synch(self, x):
        xx = np.ascontiguousarray(x)
        self.mpi_comm.Bcast([xx, MPI.DOUBLE], root=0)
        x[:] = xx

    def _grads(self, x):
        if self.mpi_comm.rank==0:
            self.mpi_comm.Bcast([np.int32(3), MPI.INT])
        #synchroize across all mpi nodes
        self.synch(x)
        g = super(ColDeepMPI, self)._grads(x)
        g_all = g.copy()
        self.mpi_comm.Reduce([g, MPI.DOUBLE], [g_all, MPI.DOUBLE], root=0)
        return g_all

    def _objective(self, x):
        if self.mpi_comm.rank==0:
            self.mpi_comm.Bcast([np.int32(2), MPI.INT])
        self.synch(x)
        o = super(ColDeepMPI, self)._objective(x)
        o_all = np.zeros(1)
        self.mpi_comm.Reduce([np.float64(o), MPI.DOUBLE], [o_all, MPI.DOUBLE], root=0)
        return o_all

    def _objective_grads(self, x):
        if self.mpi_comm.rank==0:
            self.mpi_comm.Bcast([np.int32(1), MPI.INT])
        self.synch(x)
        o, g = super(ColDeepMPI, self)._objective_grads(x)
        g_all = g.copy()
        o_all = np.zeros(1)
        self.mpi_comm.Reduce([g, MPI.DOUBLE], [g_all, MPI.DOUBLE], root=0)
        self.mpi_comm.Reduce([np.float64(o), MPI.DOUBLE], [o_all, MPI.DOUBLE], root=0)
        return o_all, g_all

    def optimize(self,*a, **kw):
        if self.mpi_comm.rank==0:
            super(ColDeepMPI, self).optimize(*a, **kw)
            self.mpi_comm.Bcast([np.int32(-1), MPI.INT])#after optimization , tell all the mpi processes to exit
        else:
            x = self.optimizer_array.copy()
            while True:
                flag = np.zeros(1,dtype=np.int32)
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    self._objective_grads(x)
                elif flag==2:
                    self._objective(x)
                elif flag==3:
                    self._grads(x)
                elif flag==-1:
                    break
                else:
                    raise ValueError, "bad integer broadcast"

#TODO: move this helper to its own utility file
def divide_data(datanum, comm):
    residue = (datanum)%comm.size
    datanum_list = np.empty((comm.size),dtype=np.int32)
    for i in xrange(comm.size):
        if i<residue:
            datanum_list[i] = int(datanum/comm.size)+1
        else:
            datanum_list[i] = int(datanum/comm.size)
    if comm.rank<residue:
        size = datanum/comm.size+1
        offset = size*comm.rank
    else:
        size = datanum/comm.size
        offset = size*comm.rank+residue
    return offset, offset+size, datanum_list


def build_supervised(X, Y, Qs, Ms, ranks=None, mpi=False, ARD_X=False, useGPU=False, stochastic=False, S_param='chol'):
    """
    Build a coldeep structure with len(Qs) hidden layers.

    Note that len(Ms) = len(Qs) + 1, since there's always 1 more GP than there
    are hidden layers.

    ranks are optionally a list of the rank of the approximation to the covariance
    """
    Ms, Qs = np.asarray(Ms), np.asarray(Qs)
    assert len(Ms) == (1 + len(Qs))
    if ranks:
        assert len(ranks)==len(Ms)

    Nx,D_in = X.shape
    Ny, D_out = Y.shape
    assert Nx==Ny

    if mpi:
        assert not stochastic, "mpi and stochastic not compatible right now"
        #cut the data into chunks
        start, stop, _ = divide_data(X.shape[0], MPI.COMM_WORLD)
        Xinit = X[start:stop]
        Yinit = Y[start:stop]
    elif stochastic:
        Xinit = X[:2] #arbitrary 2 points for init
        Yinit = Y[:2]
    else:
        Xinit=X
        Yinit=Y

    layers = []
    #input layer
    layers.append(InputLayerFixed(X=Xinit,
                      input_dim=D_in,
                      output_dim=Qs[0],
                      kern=GPy.kern.RBF(D_in, ARD=ARD_X, useGPU=useGPU),
                      Z=np.random.randn(Ms[0], D_in),
                      beta=500.,
                      S_param=S_param,
                      name='layerX'))


    #hidden layers
    for h in range(len(Qs)-1):
        layers.append(HiddenLayer(input_dim=Qs[h],
            output_dim=Qs[h+1],
            kern=GPy.kern.RBF(Qs[h], ARD=True, useGPU=useGPU),
            Z=np.random.randn(Ms[h+1], Qs[h]),
            beta=500.,
            S_param=S_param,
            name='layer%i'%h))
        layers[-2].add_layer(layers[-1])

    #output layer
    layers.append(ObservedLayer(Y=Yinit,
        input_dim=Qs[-1],
        output_dim=D_out,
        kern=GPy.kern.RBF(Qs[-1], ARD=True, useGPU=useGPU),
        Z=np.random.randn(Ms[-1], Qs[-1]),
        beta=500.,
        S_param=S_param,
        name='layerY'))
    layers[-2].add_layer(layers[-1])

    if ranks:
        i = indexes_to_fix_for_low_rank(ranks[0], Ms[0])
        for ii in i:
            layers[0].q_of_U_choleskies[ii].constrain_fixed(trigger_parent=False)
        for h in range(len(Qs)-1):
            i = indexes_to_fix_for_low_rank(ranks[h+1], Ms[h+1])
            for ii in i:
                layers[h+1].q_of_U_choleskies[ii].constrain_fixed(trigger_parent=False)
        i = indexes_to_fix_for_low_rank(ranks[-1], Ms[-1])
        for ii in i:
            layers[-1].q_of_U_choleskies[ii].constrain_fixed(trigger_parent=False)



    if mpi:
        assert not stochastic, "mpi and stochastic not compatible right now"
        m = ColDeepMPI(layers, name='deep_gp', comm=MPI.COMM_WORLD)
        for layer in layers:
            layer.KL_scaling = 1./MPI.COMM_WORLD.size

        #synchronize
        param = m[:]
        MPI.COMM_WORLD.Bcast([param, MPI.DOUBLE], root=0)
        m[:] = param
    elif stochastic:
        m = ColDeepStochastic(layers, X, Y, name='deep_gp')
    else:
        m = ColDeep(layers, name='deep_gp')
    return m


