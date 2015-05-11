import GPy
import numpy as np
import GPy.util.choleskies
import plotting
from special_einsum import special_einsum

class Layer(GPy.core.parameterization.Parameterized):
    """
    A general Layer class, the base for hidden, input and output layers.
    """
    def __init__(self, input_dim, output_dim, kern, Z, beta=10.0, natgrads=False, S_param='chol', name='layer'):
        super(Layer, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = Z.shape[0]

        #a factor by which to multiply the KL (only used in parallel implementations)
        self.KL_scaling = 1.

        #store Z, kern, beta in this Parameterized object.
        assert Z.shape[1] == self.input_dim
        self.kern = kern
        self.Z = GPy.core.Param('Z', Z)
        self.beta = GPy.core.Param('beta', beta, GPy.core.parameterization.transformations.Logexp())
        self.link_parameters(self.Z, self.kern, self.beta)

        self.natgrads = natgrads
        # initialize q(U)
        #make the mean a random draw from I
        if not self.natgrads:
            self.q_of_U_mean = GPy.core.Param('q(U)_mean', np.random.randn(self.num_inducing, self.output_dim))
            self.link_parameter(self.q_of_U_mean)
            #make the mean a random draw from Kmm
            #self.q_of_U_mean = GPy.core.Param('q(U)_mean', np.random.multivariate_normal(np.zeros(self.num_inducing), self.kern.K(self.Z), self.output_dim).T)

            self.S_param = S_param
            if S_param=='chol':
                chols = GPy.util.choleskies.triang_to_flat(np.dstack([np.eye(self.num_inducing)*0.1 for i in range(self.output_dim)]))
                self.q_of_U_choleskies = GPy.core.Param('q(U)_chol', chols)
                self.link_parameter(self.q_of_U_choleskies)
            elif S_param=='diag':
                self.q_of_U_diags = GPy.core.Param('q(U)_diag',np.ones((self.num_inducing, self.output_dim)),GPy.core.parameterization.transformations.Logexp())
                self.link_parameter(self.q_of_U_diags)
            else:
                raise NotImplementedError

        else:
            #initialize using the natural gradient method
            mean = np.random.randn(self.num_inducing, self.output_dim)
            precision = np.dstack([np.eye(self.num_inducing)*10 for i in range(self.output_dim)])
            Sim = np.einsum('ijk,jk->ik', precision, mean)
            self.set_vb_param(np.hstack((Sim.flatten(), -0.5*precision.flatten() )))

        #and empty list to contain the lower layers
        self.lower_layers = []

    def set_vb_param(self, p):
        Sim = p[:self.num_inducing*self.output_dim].reshape(self.num_inducing, self.output_dim)
        prec = -2*p[self.num_inducing*self.output_dim:].reshape(self.num_inducing, self.num_inducing, self.output_dim)
        covs, _, Ls, logdets = zip(*[GPy.util.linalg.pdinv(prec[:,:,i]) for i in range(self.output_dim)])
        self.q_of_U_covariance = np.dstack(covs)
        self.q_of_U_precision= prec
        self.q_of_U_mean = np.einsum('ijk,jk->ik', covs, Sim)
        self.q_of_U_cov_logdet = -np.array(logdets)

        #don't forget to do a forward/backwards pass to update teh gradients!

    def grad_natgrad(self):
        #return the gradient and natural gradient in the parameters of q(U)
        pass #TODO


    def initialize_inducing(self, X):
        """
        iniitalize the inducing input points from the array X, and recurse to lower layers
        """
        i = np.random.permutation(X.shape[0])[:self.num_inducing]
        self.Z[:] = X[i]
        self.compute_output_dist()
        [l.initialize_inducing(self.q_of_X_out.mean*1) for l in self.lower_layers]

    def predict(self, Xtest, full_cov=False, noise_off=False):
        """
        Given the posterior q(U), the kernel and some test points Xtest,
        compute the Gaussian posterior for this layer
        """
        Knm = self.kern.K(Xtest, self.Z)
        mu = np.dot(Knm, self.Kmmi).dot(self.q_of_U_mean)
        if self.woodbury_inv is None:
            if self.S_param is 'chol':
                self.woodbury_inv = self.Kmmi[:,:,None] - np.einsum('ij,jkl,km->iml', self.Kmmi, self.q_of_U_covariance, self.Kmmi)
            else:
                self.woodbury_inv = self.Kmmi[:,:,None] - np.einsum('ijk,jl->ilk', self.Kmmi[:,:,None]*self.q_of_U_diags[None,:,:], self.Kmmi)

        if full_cov:
            Knn = self.kern.K(Xtest)
            var = Knn[:,:,None] - np.einsum('ij, jkl,mk->iml', Knm, self.woodbury_inv, Knm)
            if not noise_off:
                var += np.eye(Xtest.shape[0])[:,:,None]/self.beta
        else:
            Knn = self.kern.Kdiag(Xtest)
            var = Knn[:,None] - np.einsum('ij,jkl,ik->il', Knm, self.woodbury_inv, Knm)
            if not noise_off:
                var += 1./self.beta
        return mu, var

    def posterior_samples(self, Xtest, full_cov=False, noise_off=False):
        """
        Produce samples from the posterior at the points Xtest.
        
        if not full_cov, we produce a separate draw for each point in Xtest. Otherwise, a single multivariate (correlated) draw.
        """
        m,v = self.predict(Xtest, full_cov=full_cov, noise_off=noise_off)
        if full_cov:
            return np.vstack([np.random.multivariate_normal(m[:,i], v[:,:,i]) for i in range(m.shape[1])]).T
        else:
            return m + np.random.randn(*m.shape)*np.sqrt(v)

    def add_layer(self, layer):
        """
        Simply append a layer onto the list of lower layers
        """
        assert layer.input_dim == self.output_dim
        self.lower_layers.append(layer)
        layer.previous_layer = self

    def shared_computations(self):
        #essential computations
        self.Kmmi, Lm, Lmi, self.log_det_Kmm = GPy.util.linalg.pdinv(self.Kmm)
        self.psi1Kmmi = np.dot(self.psi1, self.Kmmi)
        if not self.natgrads:
            if self.S_param is 'chol':
                L = GPy.util.choleskies.flat_to_triang(self.q_of_U_choleskies)
                self.q_of_U_covariance = np.einsum('ijk,ljk->ilk', L, L)
                self.q_of_U_precision = GPy.util.choleskies.multiple_dpotri(L)
                self.q_of_U_cov_logdet = 2.*np.array([np.sum(np.log(np.abs(np.diag(L[:,:,i])))) for i in range(self.output_dim)])
                uuT = np.dot(self.q_of_U_mean, self.q_of_U_mean.T) + self.q_of_U_covariance.sum(-1)
                self.psi1KmmiS = np.einsum('ij,jkl->ikl', self.psi1Kmmi, self.q_of_U_covariance) # intermediate computation
            elif self.S_param is 'diag':
                self.q_of_U_cov_logdet = np.sum(np.log(self.q_of_U_diags),0)
                uuT = np.dot(self.q_of_U_mean, self.q_of_U_mean.T) + np.diag(self.q_of_U_diags.sum(-1))
                self.q_of_U_precision = np.dstack([np.diag(1./x) for x in 1*self.q_of_U_diags.T])
                self.psi1KmmiS = self.psi1Kmmi[:,:,None]*self.q_of_U_diags[None,:,:]



        self.KiuuT = np.dot(self.Kmmi, uuT)
        self.KiuuTKi = self.KiuuT.dot(self.Kmmi)
        self.KmmiPsi2 = np.dot(self.Kmmi, self.psi2)
        self.KmmiPsi2Kmmi = self.KmmiPsi2.dot(self.Kmmi)

        #this thing is only used for prediction
        self.woodbury_inv = None

    def compute_trace_term(self):
        trace = -0.5*self.output_dim*(self.psi0.sum() - np.trace(self.KmmiPsi2))
        self._log_marginal_contribution += self.beta*trace
        self.beta.gradient += trace
        self.dL_dpsi0 += -0.5*self.beta*self.output_dim
        self.dL_dpsi2 += 0.5*self.beta*self.Kmmi*self.output_dim
        self.dL_dKmm  += -0.5*self.beta*self.KmmiPsi2Kmmi*self.output_dim

    def compute_KL_term(self):
        """Kullback Leibler term KL[q(u)||p(u)]"""
        self._log_marginal_contribution -= self.KL_scaling * 0.5*(self.output_dim * self.log_det_Kmm  - self.q_of_U_cov_logdet.sum() + self.num_inducing*self.output_dim + np.trace(self.KiuuT))
        self.dL_dKmm += self.KL_scaling * 0.5*(-self.Kmmi*self.output_dim + self.KiuuTKi)
        self.dL_duuT += self.KL_scaling * 0.5*(self.q_of_U_precision - self.Kmmi[:,:,None])
        self.dL_dEu -= self.KL_scaling * np.einsum('ijk,jk->ik', self.q_of_U_precision, self.q_of_U_mean)

    def gradient_updates(self):
        """set the derivatives in the kernel and in Z"""
        self.kern.update_gradients_full(self.dL_dKmm, self.Z)
        g = self.kern._gradient_array_.copy()
#         self.dL_dpsi2 = np.repeat(self.dL_dpsi2[None,:,:], self.q_of_X_in.shape[0], axis=0)
        self.kern.update_gradients_expectations(Z=self.Z, variational_posterior=self.q_of_X_in, dL_dpsi0=self.dL_dpsi0, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2)
        self.kern._gradient_array_ += g

        self.Z.gradient = self.kern.gradients_X(self.dL_dKmm, self.Z)
        self.Z.gradient += self.kern.gradients_Z_expectations(Z=self.Z, variational_posterior=self.q_of_X_in, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2, dL_dpsi0=self.dL_dpsi0)

        if not self.natgrads:
            self.q_of_U_mean.gradient = self.dL_dEu + 2.*np.einsum('ijk,jk->ik', self.dL_duuT, self.q_of_U_mean)
            if self.S_param is 'chol':
                L = GPy.util.choleskies.flat_to_triang(self.q_of_U_choleskies)
                dL_dchol = 2.*np.einsum('ijk,jlk->ilk', self.dL_duuT, L)
                self.q_of_U_choleskies.gradient = GPy.util.choleskies.triang_to_flat(dL_dchol)
            else:
                self.q_of_U_diags.gradient = np.vstack([np.diag(self.dL_duuT[:,:,i]) for i in xrange(self.output_dim)]).T

    def compute_output_dist(self):
        """Gaussian distributions to pass forward"""
        forward_mean = self.psi1Kmmi.dot(self.q_of_U_mean) # assumes mean is MxD
        if self.S_param is 'chol':
            #forward_var = np.einsum("ij,jkl,ki->il", self.psi1Kmmi, self.q_of_U_covariance, self.psi1Kmmi.T) + 1./self.beta # assumes covariance is MMD
            forward_var = np.einsum('ijk,ij->ik', self.psi1KmmiS, self.psi1Kmmi) + 1./self.beta
        else:
            forward_var = np.square(self.psi1Kmmi).dot(self.q_of_U_diags) + 1./self.beta#TODO: check this

        self.q_of_X_out = GPy.core.parameterization.variational.NormalPosterior(forward_mean, forward_var)

    def backpropagated_gradients(self, dL_dmean, dL_dvar):
        """Given derivatives of terms in lower layers, update derivatives in this layer"""
        Kim = self.Kmmi.dot(self.q_of_U_mean)

        #additional gradients for self.psi1 from the chain rule
        self.dL_dpsi1 += np.einsum('ij,kj->ik', dL_dmean, Kim)
        tmp0 = np.einsum('kjl,kl->kj', self.psi1KmmiS, dL_dvar)
        self.dL_dpsi1 += tmp0.dot(self.Kmmi) * 2.

        #additional gradients for self.Kmm from the chain rule
        tmp1 = np.dot(self.psi1Kmmi.T, dL_dmean)
        tmp2 = np.dot(Kim, tmp1.T)
        self.dL_dKmm += -0.5*(tmp2 + tmp2.T)
        tmp3 = tmp0.T.dot(self.psi1)
        self.dL_dKmm += -np.dot(self.Kmmi, tmp3 + tmp3.T).dot(self.Kmmi)

        #additional gradients for self.beta from the chain rule
        self.beta.gradient += -dL_dvar.sum()/self.beta**2

        #additional gradients for q(U)
        self.dL_dEu += tmp1
        dL_dS = special_einsum(self.psi1Kmmi, dL_dvar)
        self.dL_duuT += dL_dS
        self.dL_dEu -= 2.*np.einsum('ijk,jk->ik', dL_dS, self.q_of_U_mean)

    def reset_gradients(self):
        self._log_marginal_contribution = 0.
        self.dL_dpsi0 = np.zeros_like(self.psi0)
        self.dL_dpsi1 = np.zeros_like(self.psi1)
        self.dL_dpsi2 = np.zeros_like(self.psi2)
        self.dL_dKmm  = np.zeros_like(self.Kmm)
        self.beta.gradient = 0.
        self.dL_duuT = np.zeros((self.num_inducing, self.num_inducing, self.output_dim))
        self.dL_dEu = np.zeros((self.num_inducing, self.output_dim))




class HiddenLayer(Layer):

    def plot(self):
        plotting.plot_hidden_layer(self)

    def log_density_sampling(self, X, Y):
        XX = self.posterior_samples(X)
        return self.lower_layers[0].log_density_sampling(XX, Y)

    def predict_forward_sampling(self, X, correlated=False, noise_off=False):
        XX = self.posterior_samples(X, full_cov=correlated, noise_off=noise_off)
        return self.lower_layers[0].predict_forward_sampling(XX, correlated=correlated, noise_off=noise_off)

    def predict_means(self, X):
        m, v = self.predict(X)
        return self.lower_layers[0].predict_means(m)


    def feed_forward(self, q_of_X_in):
        """
        Compute the distribution for the outputs of this layer, as well as any
        marginal likelihood terms that occur
        """

        #store the distribution from the incoming layer
        self.q_of_X_in = q_of_X_in

        #kernel computations
        self.psi0 = self.kern.psi0(self.Z, q_of_X_in)
        self.psi1 = self.kern.psi1(self.Z, q_of_X_in)
        self.psi2 = self.kern.psi2(self.Z, q_of_X_in)
        self.Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*1e-3

        self.reset_gradients()
        self.shared_computations()
        self.compute_trace_term()
        self.compute_KL_term()

        #complete the square term
        psi2_centered = self.psi2 - np.dot(self.psi1.T, self.psi1)
        tmp = np.sum(psi2_centered * self.KiuuTKi)
        self.beta.gradient += -0.5*tmp
        self._log_marginal_contribution += -0.5*self.beta*tmp
        self.dL_dpsi1 += self.beta*self.KiuuTKi.dot(self.psi1.T).T
        self.dL_dpsi2 += -0.5*self.beta*self.KiuuTKi
        tmp = self.KiuuTKi.dot(psi2_centered).dot(self.Kmmi)
        self.dL_dKmm += 0.5*self.beta*(tmp + tmp.T)
        self.dL_duuT += -0.5*self.beta*(self.Kmmi.dot(psi2_centered).dot(self.Kmmi))[:,:,None]

        self.compute_output_dist()

        #feed forward to downstream layers
        [l.feed_forward(self.q_of_X_out) for l in self.lower_layers]

    def feed_backwards(self, dL_dmean, dL_dvar):
        """
        Layers ahead of this one will compute terms that affect the marginal
        likelihood. Given the derivative of those terms with respect to what
        this layer has fed forward, we can compute the derivatives wrt
        parameters in this layer.
        """

        self.backpropagated_gradients(dL_dmean, dL_dvar)

        self.gradient_updates()

        #compute the gradients wrt the input q(X) to feed backward to the previous layer
        X_mean_grad, X_var_grad = self.kern.gradients_qX_expectations(variational_posterior=self.q_of_X_in, Z=self.Z, dL_dpsi0=self.dL_dpsi0, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2)
        self.previous_layer.feed_backwards(X_mean_grad, X_var_grad)

class InputLayerFixed(Layer):
    def __init__(self, X, input_dim, output_dim, kern, Z, beta=0.01, S_param='chol', name='input_layer'):
        super(InputLayerFixed, self).__init__(input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta, S_param=S_param, name=name)
        assert X.shape[1] == self.input_dim
        if isinstance(X, (GPy.core.parameterization.ObsAr, GPy.core.parameterization.variational.VariationalPosterior)):
            self.X = X.copy()
        else: self.X = GPy.core.parameterization.ObsAr(X)

    def log_density_sampling(self, X, Y):
        XX = self.posterior_samples(X)
        return self.lower_layers[0].log_density_sampling(XX, Y)

    def predict_forward_sampling(self, X, correlated=False, noise_off=False):
        XX = self.posterior_samples(X, full_cov=correlated, noise_off=noise_off)
        return self.lower_layers[0].predict_forward_sampling(XX, correlated=correlated, noise_off=noise_off)

    def predict_means(self, X):
        m, v = self.predict(X)
        return self.lower_layers[0].predict_means(m)

    def plot(self):
        plotting.plot_input_layer(self)


    def feed_forward(self):
        """
        Compute the distribution for the outputs of this layer, as well as any
        marginal likelihood terms that occur.
        """

        #kernel computations
        self.psi0 = self.kern.Kdiag(self.X)
        self.psi1 = self.kern.K(self.X, self.Z)
        self.psi2 = self.psi1.T.dot(self.psi1)
        self.Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*1e-3

        self.reset_gradients()
        self.shared_computations()
        self.compute_trace_term()
        self.compute_KL_term()
        self.compute_output_dist()

        #feed forward to downstream layers
        [l.feed_forward(self.q_of_X_out) for l in self.lower_layers]

    def feed_backwards(self, dL_dmean, dL_dvar):
        """
        Layers ahead of this one will compute terms that affect the marginal
        likelihood. Given the derivative of those terms with respect to what
        this layer has fed forward, we can compute the derivatives wrt
        parameters in this layer.
        """

        self.backpropagated_gradients(dL_dmean, dL_dvar)
        self.gradient_updates()

    def gradient_updates(self):
        #note that the kerel gradients are a little different because there's no q(X), just a fixed X
        self.kern.update_gradients_full(self.dL_dKmm, self.Z)
        g = self.kern._gradient_array_.copy()
        dL_dKnm = self.dL_dpsi1 + 2.*self.psi1.dot(self.dL_dpsi2)
        self.kern.update_gradients_full(dL_dKnm, self.X, self.Z)
        g += self.kern._gradient_array_.copy()
        self.kern.update_gradients_diag(self.dL_dpsi0, self.X)
        self.kern._gradient_array_ += g

        self.Z.gradient = self.kern.gradients_X(self.dL_dKmm, self.Z)
        self.Z.gradient += self.kern.gradients_X(dL_dKnm.T, self.Z, self.X)

        self.q_of_U_mean.gradient = self.dL_dEu + 2.*np.einsum('ijk,jk->ik',self.dL_duuT, self.q_of_U_mean)
        if self.S_param is 'chol':
            L = GPy.util.choleskies.flat_to_triang(self.q_of_U_choleskies)
            dL_dchol = 2.*np.einsum('ijk,jlk->ilk', self.dL_duuT, L)
            self.q_of_U_choleskies.gradient = GPy.util.choleskies.triang_to_flat(dL_dchol)
        else:
            self.q_of_U_diags.gradient = np.vstack([np.diag(self.dL_duuT[:,:,i]) for i in xrange(self.output_dim)]).T



class ObservedLayer(Layer):
    def plot(self):
        plotting.plot_output_layer(self)

    def __init__(self, Y, input_dim, output_dim, kern, Z, beta=0.01, S_param='chol', name='output_layer'):
        super(ObservedLayer, self).__init__(input_dim, output_dim, kern, Z, beta=beta, S_param=S_param, name=name)
        assert Y.shape[1] == output_dim
        self.Y = Y
        tmp = np.random.randn(self.num_inducing, self.num_inducing)

    def log_density_sampling(self, X, Y):
        m, v = self.predict(X)
        m, v = m.reshape(-1,*Y.shape, order='F'), v.reshape(-1,*Y.shape, order='F')
        logdensities =  -0.5*np.log(2*np.pi) - 0.5*np.log(v) -0.5*np.square(m - Y[None,:,:])/v
        logdensities = logdensities.sum(-1)#sum over the output dimensions
        #safe exponentiate and average over the samples
        Nsamples = logdensities.shape[0]
        from scipy.misc import logsumexp
        return logsumexp(logdensities, axis=0) - np.log(Nsamples)

    def predict_forward_sampling(self, X, correlated=False, noise_off=False):
        XX = self.posterior_samples(X, full_cov=correlated, noise_off=noise_off)
        return XX

    def predict_means(self, X):
        return self.predict(X)


    def feed_forward(self, q_of_X_in):
        #store the distribution from the incoming layer
        self.q_of_X_in = q_of_X_in

        #kernel computations
        self.psi0 = self.kern.psi0(self.Z, q_of_X_in)
        self.psi1 = self.kern.psi1(self.Z, q_of_X_in)
        self.psi2 = self.kern.psi2(self.Z, q_of_X_in)
        self.Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*1e-3
        self.Kmmi, Lm, Lmi, self.log_det_Kmm = GPy.util.linalg.pdinv(self.Kmm)


        self.reset_gradients()
        self.shared_computations()
        self.compute_trace_term()
        self.compute_KL_term()

        #data likelihood terms
        if self.S_param is 'chol':
            uuT = np.dot(self.q_of_U_mean, self.q_of_U_mean.T) + self.q_of_U_covariance.sum(-1)
        else:
            uuT = np.dot(self.q_of_U_mean, self.q_of_U_mean.T) + np.diag(self.q_of_U_diags.sum(-1))
        proj_mean = self.psi1Kmmi.dot(self.q_of_U_mean)
        muY = np.dot(self.q_of_U_mean, self.Y.T)
        N = q_of_X_in.shape[0]
        expected_dist = np.sum(np.square(self.Y)) + np.sum(uuT * self.KmmiPsi2Kmmi) - 2.*np.sum(self.Y*proj_mean)
        self._log_marginal_contribution += -0.5*N*self.output_dim*np.log(2*np.pi)
        self._log_marginal_contribution += 0.5*N*self.output_dim*np.log(self.beta)
        self._log_marginal_contribution += -0.5*self.beta*expected_dist
        self.dL_dpsi1 += self.beta*np.dot(self.Kmmi, muY).T
        self.dL_dpsi2 += -0.5*self.beta*self.Kmmi.dot(uuT).dot(self.Kmmi)
        tmp = self.Kmmi.dot(uuT.dot(self.Kmmi).dot(self.psi2) - muY.dot(self.psi1)).dot(self.Kmmi) #TODO: inefficient?
        self.dL_dKmm += 0.5*self.beta*(tmp + tmp.T)
        self.beta.gradient += 0.5*N*self.output_dim/self.beta -0.5*expected_dist
        self.dL_duuT += -0.5*self.beta*self.KmmiPsi2Kmmi[:,:,None]
        self.dL_dEu += self.beta*np.einsum('ij,ik->jk', self.psi1Kmmi, self.Y)

        #since we're the leaf of the tree, update all the gradients now.
        self.gradient_updates()

        #compute the gradients wrt the input q(X) to feed backward to the previous layer
        X_mean_grad, X_var_grad = self.kern.gradients_qX_expectations(variational_posterior=self.q_of_X_in, Z=self.Z, dL_dpsi0=self.dL_dpsi0, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2)

        self.previous_layer.feed_backwards(X_mean_grad, X_var_grad)




class InputLayer(GPy.core.parameterization.Parameterized):
    """
    A simple layer to represent the unsupervised inputs to a deep network
    """
    def __init__(self, X_mean, X_variance, name='input_layer'):
        super(InputLayer, self).__init__(name)

        self.X = GPy.core.parameterization.variational.NormalPosterior(X_mean, X_variance)
        self.link_parameter(self.X)

        self.variational_prior = GPy.core.parameterization.variational.NormalPrior()
        self.lower_layers = []

    def add_layer(self, layer):
        assert layer.input_dim == self.X.shape[1]
        self.lower_layers.append(layer)
        layer.previous_layer = self

    def feed_forward(self):
        [l.feed_forward(self.X) for l in self.lower_layers]

    def feed_backwards(self, dL_dmean, dL_dvar):
        self._log_marginal_contribution = -self.variational_prior.KL_divergence(self.X)
        self.X.mean.gradient, self.X.variance.gradient = dL_dmean, dL_dvar
        self.variational_prior.update_gradients_KL(self.X)


