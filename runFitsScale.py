import os
import h5py
import multiprocessing

ncpu = multiprocessing.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, hist2array, fill_hist

import matplotlib.pyplot as plt
import mplhep as hep
# matplotlib stuff
plt.style.use([hep.style.ROOT])
# hep.cms.label(loc=0, year=2016, lumi=35.9, data=True)
# hep.cms.text('Simulation')

from fittingFunctionsBinned import scaleFromModelPars, splitTransformPars, nllBinsFromBinPars, chi2LBins, scaleSqSigmaSqFromBinsPars,scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars, nllBinsFromBinParsRes, plotsSingleMu, scaleSqFromModelParsSingleMu, sigmaSqFromModelParsSingleMu, nllBinsFromSignalBinPars
from obsminimization import pmin
import argparse
import functools
import time
import sys

#slower but lower memory usage calculation of hessian which
#explicitly loops over hessian rows
def hessianlowmem(fun):
    def _hessianlowmem(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianlowmem, f=fun)

#compromise version which vectorizes the calculation, but only partly to save memory
def hessianoptsplit(fun, vsize=4):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        n = np.prod(x.shape)
        idxs =  np.arange(vsize, n, vsize)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        splitbasis = np.split(basis,idxs)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        return np.concatenate([vhvp(b) for b in splitbasis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)

#optimized version which is faster than the built-in hessian for some reason
# **TODO** follow up with jax authors to understand why
def hessianopt(fun):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return vhvp(basis).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)


#def vgrad(fun):
    #g = jax.grad(fun)
    #return jax.vmap(g)

def hvp(fun):
    def _hvp(x, v, f):
        return jax.jvp(jax.grad(f), (x,), (v,))[1]
    return functools.partial(_hvp, f=fun)

class CachingHVP():
    def __init__(self,fun):
        self.grad = jax.jit(jax.grad(fun))
        self.x = None
        self.flin = None
        
    def hvp(self,x,v):
        if self.x is None or not np.equal(x,self.x).all():
            _,flin = jax.linearize(self.grad,x)
            self.flin = jax.jit(flin)
            #self.flin = flin
            self.x = x
        return self.flin(v)
    
#def blockfgrad(fun):
    #g = jax.grad(fun)
    #return jax.jit(jax.vmap(g))
    
class CachingBlockGrads():
    def __init__(self,fun,nblocks,static_argnums=None,):
        hess = jax.hessian(fun)
        vhess = jax.vmap(hess)
        self._vhess = jax.jit(vhess,static_argnums=static_argnums)
        self._vmatmul = jax.jit(jax.vmap(np.matmul))
        self._fgrad = jax.jit(jax.vmap(jax.value_and_grad(fun)),static_argnums=static_argnums)
        self.vhessres = None
        self.x = None
        self.nblocks = nblocks
        
    def hvp(self,x,v, *args):
        if self.x is None or not np.equal(x,self.x).all():
            self.vhessres = self._vhess(x.reshape((self.nblocks,-1)),*args)
            self.x = x
        return self._vmatmul(self.vhessres,v.reshape((self.nblocks,-1))).reshape((-1,))
    
    def fgrad(self, x, *args):
        f,g = self._fgrad(x.reshape((self.nblocks,-1)), *args)
        f = np.sum(f, axis=0)
        g = g.reshape((-1,))
        return f,g
   
#wrapper to handle printing which otherwise doesn't work properly from bfgs apparently
class NLLHandler():
    def __init__(self, fun, fundebug = None):
        self.fun = fun
        self.fundebug = fundebug
        self.iiter = 0
        self.f = 0.
        self.grad = np.array(0.)
        
    def wrapper(self, x, *args):
        f,grad = self.fun(x,*args)
        if np.isnan(f) or np.any(np.isnan(grad)):
            print("nan detected")
            print(x)
            print(f)
            print(grad)
            
            if self.fundebug is not None:
                self.fundebug(x)
                
            assert(0)    
        self.f = f
        self.grad = grad
        return f,grad
    
    def callback(self,x):
        print(self.iiter, self.f, np.linalg.norm(self.grad))
        self.iiter += 1

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-isData', '--isData', default=False, action='store_true', help='Use to run on data')

args = parser.parse_args()
isJ = args.isJ
isData = args.isData

if isJ:
    fileJ = open("calInputJ{}_48etaBins_5ptBins.pkl".format("DATA" if isData else "MC"), "rb")
    pkgJ = pickle.load(fileJ)

    datasetJ = pkgJ['dataset']
    etas = pkgJ['edges'][0]
    ptsJ = pkgJ['edges'][2]
    massesJ = pkgJ['edges'][-1]
    good_idxJ = pkgJ['good_idx']
    dataset = datasetJ
    pts = ptsJ
    masses = massesJ
    good_idx= good_idxJ

    filegen = open("calInput{}MCgen_48etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
    datasetgen = pickle.load(filegen)

nEtaBins = len(etas)-1
nPtBins = len(pts)-1
nBins = dataset.shape[0]

print(pts)
print(nBins)

scale = np.ones((nBins,),dtype='float64')
sigma = 1e-2*np.ones((nBins,),dtype='float64')
fbkg = 0.05*np.ones((nBins,),dtype='float64')
slope = 0.02*np.ones((nBins,),dtype='float64')

# plotsBkg(scale,sigma,fbkg,slope,dataset,datasetgen,masses,isJ,etas, good_idx)

xscale = np.stack([scale,sigma,fbkg,slope],axis=-1)
xscale = np.zeros_like(xscale)
#nllBinspartial = functools.partial(nllBinsFromSignalBinPars,masses=masses)
nllBinspartial = functools.partial(nllBinsFromBinPars,masses=masses)
#parallel fit for scale, sigma, fbkg, slope in bins
xres = pmin(nllBinspartial, xscale, args=(dataset,datasetgen))
#xres = xscale
#compute covariance matrices of scale and sigma from binned fit
def hnll(x, dataset,datasetgen):
    #compute the hessian wrt internal fit parameters in each bin
    hess = jax.hessian(nllBinspartial)
    #invert to get the hessian
    cov = np.linalg.inv(hess(x,dataset,datasetgen))
    #compute the jacobian for scale and sigma squared wrt internal fit parameters
    jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
    jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
    #compute covariance matrix for scalesq and sigmasq
    covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
    #invert again to get the hessian
    hscalesigmasq = np.linalg.inv(covscalesigmasq)
    return hscalesigmasq, covscalesigmasq

fh = jax.jit(jax.vmap(hnll))
hScaleSqSigmaSqBinned, hCovScaleSqSigmaSqBinned = fh(xres,dataset,datasetgen)

fbkg, slope = bkgModelFromBinPars(xres)

scaleSqBinned, sigmaSqBinned = scaleSqSigmaSqFromBinsPars(xres)

scaleBinned = np.sqrt(scaleSqBinned)
sigmaBinned = np.sqrt(sigmaSqBinned)

scaleSqSigmaSqErrorsBinned = np.sqrt(np.diagonal(hCovScaleSqSigmaSqBinned, axis1=-1, axis2=-2))

scaleSqErrorBinned = scaleSqSigmaSqErrorsBinned[:,0]
sigmaSqErrorBinned = scaleSqSigmaSqErrorsBinned[:,1]

scaleErrorBinned = 0.5*scaleSqErrorBinned/scaleBinned
sigmaErrorBinned = 0.5*sigmaSqErrorBinned/sigmaBinned

print(scaleBinned, '+/-', scaleErrorBinned)
print(sigmaBinned, '+/-', sigmaErrorBinned)

# plotsBkg(scaleBinned,sigmaBinned,fbkg,slope,dataset,datasetgen,masses,isJ,etas, good_idx)

#have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
scaleplotBinned = ROOT.TH1D("scaleBinned", "scale", nBins, onp.linspace(0, nBins, nBins+1))
scaleplotBinned = array2hist(scaleBinned, scaleplotBinned, scaleErrorBinned)

sigmaplotBinned = ROOT.TH1D("sigmaBinned", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
sigmaplotBinned = array2hist(sigmaBinned, sigmaplotBinned, sigmaErrorBinned)

plots = [scaleplotBinned,sigmaplotBinned]

for ibin in range(nBins):
    ieta1 = good_idx[0][ibin]
    ieta2 = good_idx[1][ibin]
    ipt1 = good_idx[2][ibin]
    ipt2 = good_idx[3][ibin]
    for plot in plots:
        plot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))
for plot in plots:
    plot.GetXaxis().LabelsOption("v")

fout = ROOT.TFile("calibration{}.root".format("DATA" if isData else "MC"), 'recreate')
fout.cd()

for plot in plots:
    plot.Write()

# genmass values
massesC = 0.5*(masses[1:]+masses[:-1])
den = np.sum(datasetgen, axis=-1)
means = np.sum(massesC*datasetgen, axis=-1)/den
rms = np.std(massesC*datasetgen, axis=-1)/den
# plot parameters in a grid
pars = [scaleBinned, sigmaBinned, fbkg, slope, means, rms]
names = ["scale", "sigma", "fbkg", "slope", "genmassmean", "genmassrms"]

for i,par in enumerate(pars):
    # patch parameters
    full = onp.zeros((nEtaBins,nEtaBins,nPtBins,nPtBins),dtype='float64')
    full[good_idx] = par
    # eta1 eta2
    fig, ax1 = plt.subplots()
    ax1.set_title("{}_eta1_eta2".format(names[i]), fontsize=18)
    hep.hist2dplot(full[...,0,0],etas,etas, vmin=np.min(scaleBinned), vmax=np.max(scaleBinned))
    plt.tight_layout()
    plt.savefig("{}_eta1_eta2".format(names[i]))
    plt.clf()

    # eta1 pt1
    fig, ax1 = plt.subplots()
    ax1.set_title("{}_eta1_pt1".format(names[i]), fontsize=18)
    hep.hist2dplot(full[:,0,:,0],etas,pts, vmin=np.min(scaleBinned), vmax=np.max(scaleBinned))
    plt.tight_layout()
    plt.savefig("{}_eta1_pt1".format(names[i]))
    plt.clf()

    # pt1 pt2
    fig, ax1 = plt.subplots()
    ax1.set_title("{}_pt1_pt2".format(names[i]), fontsize=18)
    hep.hist2dplot(full[0,0,:,:],pts,pts, vmin=np.min(scaleBinned), vmax=np.max(scaleBinned))
    plt.tight_layout()
    plt.savefig("{}_pt1_pt2".format(names[i]))
    plt.clf()

with h5py.File("fits{}.hdf5".format("DATA" if isData else "MC"), mode="w") as f:
    dtype = 'float64'
    dset_scale = f.create_dataset('scale', scaleBinned.shape, dtype=dtype)
    dset_scale[...] = scaleBinned
    dset_sigma = f.create_dataset('sigma', sigmaBinned.shape, dtype=dtype)
    dset_sigma[...] = sigmaBinned
    dset_fbkg = f.create_dataset('fbkg', fbkg.shape, dtype=dtype)
    dset_fbkg[...] = fbkg
    dset_slope = f.create_dataset('slope', slope.shape, dtype=dtype)
    dset_slope[...] = slope
    dset_good_idx = f.create_dataset('good_idx', np.asarray(good_idx).shape, dtype=dtype)
    dset_good_idx[...] = np.asarray(good_idx)
    dset_etas = f.create_dataset('etas', etas.shape, dtype=dtype)
    dset_etas[...] = etas
    dset_pts = f.create_dataset('pts', pts.shape, dtype=dtype)
    dset_pts[...] = pts

