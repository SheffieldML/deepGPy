# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.plotting.matplot_dep.base_plots import gpplot, x_frame1D, x_frame2D
from matplotlib import pyplot as plt
import numpy as np

def sausage_plot(layer, Xnew, ax):
    mu, var = layer.predict(Xnew)
    gpplot(Xnew, mu, mu + 2*np.sqrt(var), mu - 2*np.sqrt(var), ax=ax)

def errorbars(layer, ax):
    if layer.S_param is 'chol':
        yerr = 2.*np.sqrt(np.vstack([np.diag(layer.q_of_U_covariance[:,:,i]) for i in range(layer.output_dim)]).T[:,0])
    else:
        yerr = 2*np.sqrt(layer.q_of_U_diag.flatten())

    ax.errorbar(layer.Z[:,0]*1, layer.q_of_U_mean[:,0]*1, yerr=yerr , linestyle='')

def plot_gaussians(q, ax,limits=None, vertical=False, color='k'):
    if limits is None:
        Xnew, xmin, xmax = x_frame1D(q.mean, resolution=2000)
    else:
        xmin, xmax = limits
        Xnew = np.linspace(xmin, xmax, 2000)[:,None]

    #compute Gaussian densities
    log_density = -0.5*np.log(2*np.pi) -0.5*np.log(q.variance) -0.5*(q.mean-Xnew.T)**2/q.variance
    density = np.exp(log_density)
    if vertical:
        [ax.plot(d, Xnew[:,0], color, linewidth=1.) for d in density]
        [ax.fill(d, Xnew[:,0], color=color, linewidth=0., alpha=0.2) for d in density]
    else:
        ax.plot(Xnew, density.T, color, linewidth=1.)
        ax.fill(Xnew, density.T, color=color, linewidth=0., alpha=0.2)


def plot_hidden_layer(layer):
    if layer.input_dim==1:
        fig = plt.figure()
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])

        Xnew, xmin, xmax = x_frame1D(np.vstack((layer.Z*1, layer.q_of_X_in.mean*1)), resolution=200)

        sausage_plot(layer, Xnew, ax1)
        errorbars(layer, ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)

        #do the gaussians for the input
        ax2 = fig.add_axes([0.2, 0.1, 0.7, 0.1], sharex=ax1)
        ax2.set_yticks([])
        plot_gaussians(layer.q_of_X_in, ax2, (xmin, xmax))
        ax2.set_xlim(xmin, xmax)
        #ax2.set_ylim(ax2.get_ylim()[::-1])

        ax3 = fig.add_axes([0.1, 0.2, 0.1, 0.7], sharey=ax1)
        plot_gaussians(layer.q_of_X_out, ax3, ax1.get_ylim(), vertical=True)
        ax3.set_xticks([])
        ax3.set_xlim(ax3.get_xlim()[::-1])

def plot_input_layer(layer):
    if layer.input_dim==1:
        fig = plt.figure()
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])

        Xnew, xmin, xmax = x_frame1D(layer.Z, resolution=200)

        sausage_plot(layer, Xnew, ax1)
        errorbars(layer, ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)

        #do crosses for the input
        ax2 = fig.add_axes([0.2, 0.1, 0.7, 0.1], sharex=ax1)
        ax2.set_yticks([])
        ax2.plot(layer.X*1, layer.X*0, 'kx', mew=2., ms=9)
        ax2.set_xlim(xmin, xmax)

        ax3 = fig.add_axes([0.1, 0.2, 0.1, 0.7], sharey=ax1)
        plot_gaussians(layer.q_of_X_out, ax3,vertical=True)
        ax3.set_xticks([])
        ax3.set_xlim(ax3.get_xlim()[::-1])

def plot_output_layer(layer):
    if layer.input_dim==1:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.2, 0.8, 0.7])

        Xnew, xmin, xmax = x_frame1D(layer.Z, resolution=200)

        sausage_plot(layer, Xnew, ax1)
        errorbars(layer, ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)

        #plot the data
        ax1.plot(layer.q_of_X_in.mean*1., layer.Y, 'kx', mew=2, ms=9 )

        #do the gaussians for the input
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.1], sharex=ax1)
        ax2.set_yticks([])
        plot_gaussians(layer.q_of_X_in, ax2, (xmin, xmax))
        ax2.set_xlim(xmin, xmax)
        #ax2.set_ylim(ax2.get_ylim()[::-1])


def plot_deep(model, xlim=None, Nsamples=0):
    if model.layerX.input_dim==1 and model.layerY.output_dim==1:
        fig, ax1 = plt.subplots(1)
        if xlim is None:
            Xnew, xmin, xmax = x_frame1D(model.layerX.X, resolution=200)
        else:
            xmin = xlim[0]
            xmax = xlim[1]
            Xnew = np.linspace(xmin, xmax, 200)[:, None]
        Xnew = np.linspace(xmin,xmax,200)[:,None]
        s = model.predict_sampling(Xnew, 1000)
        yTest = model.predict_means(Xnew)[0]
        H, xedges, yedges = np.histogram2d(np.repeat(Xnew.T,  1000, 0).flatten(), 
                                           s.flatten(), 
                                           bins=[Xnew.flatten(),
                                                 np.linspace(s.min(),s.max(),50)])
        ax1.imshow(H.T, 
                   extent=[xedges.min(), xedges.max(),
                           yedges.min(), yedges.max()], 
                   cmap=plt.cm.Blues, 
                   interpolation='nearest',
                   origin='lower',
                   aspect='auto')
        ax1.plot(model.layerX.X, model.layerY.Y, 'kx', mew=1.3)
        ax1.plot(Xnew.flatten(), yTest.flatten())
        ax1.set_ylim(yedges.min(), yedges.max())
        ax1.set_xlim(xmin, xmax)

        for n in range(Nsamples):
            Y = model.posterior_sample(Xnew)
            ax1.plot(Xnew, Y, 'r', lw=1.4)
