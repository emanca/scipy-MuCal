import os
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')
parser.add_argument('-fitMCtruth', '--fitMCtruth', default=False, action='store_true', help='Use to fit resolution parameter from mc truth')


args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration
fitMCtruth = args.fitMCtruth
"""
fileZ = open("calInputZMC_48etaBins_5ptBins.pkl", "rb")
pkgZ = pickle.load(fileZ)

datasetZ = pkgZ['dataset']
etas = pkgZ['edges'][0]
ptsZ = pkgZ['edges'][2]
massesZ = pkgZ['edges'][-1]
binCenters1Z = pkgZ['binCenters1']
binCenters2Z = pkgZ['binCenters2']
good_idxZ = pkgZ['good_idx']
"""
if isJ:
    fileJ = open("calInputJDATA_48etaBins_5ptBins.pkl", "rb")
    pkgJ = pickle.load(fileJ)

    datasetJ = pkgJ['dataset']
    etas = pkgJ['edges'][0]
    ptsJ = pkgJ['edges'][2]
    massesJ = pkgJ['edges'][-1]
    binCenters1J = pkgJ['binCenters1']
    binCenters2J = pkgJ['binCenters2']
    good_idxJ = pkgJ['good_idx']
    dataset = datasetJ
    pts = ptsJ

    masses = massesJ
    binCenters1 = binCenters1J
    binCenters2 = binCenters2J
    good_idx= good_idxJ

    filegen = open("calInput{}MCgen_48etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
    datasetgen = pickle.load(filegen)


if fitMCtruth:
    fileJ = open("calInputJMCtruth_48etaBins_20ptBins.pkl", "rb")
    pkgJtruth = pickle.load(fileJ)
    datasetJ = pkgJtruth['dataset']
    ptsJ = pkgJtruth['edges'][1]
    
    fileZ = open("calInputZMCtruth_48etaBins_80ptBins.pkl", "rb")
    pkgZtruth = pickle.load(fileZ)
    datasetZ = pkgZtruth['dataset']
    ptsZ = pkgZtruth['edges'][1]

    #merge datasets and pts
    dataset = np.concatenate((datasetJ,datasetZ), axis=0)
    pts = np.concatenate((ptsJ,ptsZ),axis=0)
    etas = pkgJtruth['edges'][0]
    masses = pkgJtruth['edges'][-1]

    good_idxJ = pkgJtruth['good_idx']
    good_idxZ = pkgZtruth['good_idx']
    total = good_idxJ + good_idxZ
    print len(total), total[0].shape,total[3].shape
    good_idx =(np.concatenate((total[0],total[2]),axis=0),np.concatenate((total[1],total[3]),axis=0))
    binCenters = np.concatenate((pkgJtruth['binCenters'],pkgZtruth['binCenters']),axis=0)
    #dataset = datasetJ
    #masses = pkgJtruth['edges'][-1]
    #good_idx = pkgJtruth['good_idx']
    #binCenters = pkgJtruth['binCenters']
    #pts = pkgJtruth['edges'][1]
    #etas = pkgJtruth['edges'][0]


nEtaBins = len(etas)-1
nPtBins = len(pts)-1
nBins = dataset.shape[0]

print(pts)
print(nBins)

scale = np.ones((nBins,),dtype='float64')
sigma = 6e-3*np.ones((nBins,),dtype='float64')
fbkg = np.zeros((nBins,),dtype='float64')
slope = np.zeros((nBins,),dtype='float64')


if fitMCtruth:
    xscale = np.stack([scale,sigma],axis=-1)
    xscale = np.zeros_like(xscale)
    nllBinspartial = functools.partial(nllBinsFromBinParsRes, masses=masses)

    #parallel fit for scale, sigma, fbkg, slope in bins
    xres = pmin(nllBinspartial, xscale, args=(dataset,))
    #xres = xscale

    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x,dataset,):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartial)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,dataset,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinned, hCovScaleSqSigmaSqBinned = fh(xres,dataset,)

else:
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

###### begin parameters fit

nModelParms = 4
A = 0.99*np.ones((nEtaBins),dtype=np.float64)
e = np.zeros((nEtaBins),dtype=np.float64)
M = 2e-5*np.ones((nEtaBins),dtype=np.float64)
a = 1e-6*np.ones((nEtaBins),dtype=np.float64)
c = 10e-9*np.ones((nEtaBins),dtype=np.float64)

xmodel = np.stack((A,M,a,c),axis=-1)

if fitMCtruth:
    chi2 = chi2LBins(xmodel, scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters,good_idx)
else:
    chi2 = chi2LBins(xmodel, scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters1, binCenters2, good_idx)

print(chi2)

if fitMCtruth:
    xmodel = pmin(chi2LBins, xmodel.flatten(), args=(scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters,good_idx), doParallel=False)
else:
    xmodel = pmin(chi2LBins, xmodel.flatten(), args=(scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx), doParallel=False)

xmodel = xmodel.reshape((-1,nModelParms))

fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
hchi2 = jax.jit(jax.hessian(chi2LBins))

if fitMCtruth:
    chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters, good_idx)
    chi2hess = hchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters, good_idx)
else:
    chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)
    chi2hess = hchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

if fitMCtruth:
    valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters, good_idx)
else:
    valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)

ndof = 2*nBins - nEtaBins*nModelParms
edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

print("nEtaBins", nEtaBins)
print("nBins", nBins)
print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

A,e,M,a,c = modelParsFromParVector(xmodel)

if fitMCtruth:
    scaleSqModel = scaleSqFromModelParsSingleMu(A, e, M, etas, binCenters, good_idx)
    sigmaSqModel = sigmaSqFromModelParsSingleMu(a, c, etas, binCenters, good_idx)
else:
    scaleSqModel = scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx)
    sigmaSqModel = sigmaSqFromModelPars(a, c, etas, binCenters1, binCenters2, good_idx)


scaleModel = np.sqrt(scaleSqModel)
sigmaModel = np.sqrt(sigmaSqModel)

print(xmodel, "+/-", errsmodel)
print(edm, "edm")

diag = np.diag(np.sqrt(np.diag(invhess)))
diag = np.linalg.inv(diag)
corr = np.dot(diag,invhess).dot(diag)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.clf()
plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("corrMC.pdf")

#if fitMCtruth:
    #plotsSingleMu(scaleBinned,sigmaBinned,dataset,masses)
#else:
    #plotsBkg(scaleBinned,sigmaBinned,fbkg,slope,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)

print("computing scales and errors:")

ndata = np.sum(dataset,axis=-1)

Aerr = errsmodel[:,0]
eerr = errsmodel[:,1]
Merr = errsmodel[:,2]
Werr = errsmodel[:,3]
aerr = errsmodel[:,4]
cerr = errsmodel[:,5]
berr = errsmodel[:,6]
derr = errsmodel[:,7]

etaarr = onp.array(etas.tolist())
hA = ROOT.TH1D("A", "A", nEtaBins, etaarr)
he = ROOT.TH1D("e", "e", nEtaBins, etaarr)
hM = ROOT.TH1D("M", "M", nEtaBins, etaarr)
ha = ROOT.TH1D("a", "a", nEtaBins, etaarr)
hc = ROOT.TH1D("c", "c", nEtaBins, etaarr)

hA = array2hist(A, hA, Aerr)
he = array2hist(e, he, eerr)
hM = array2hist(M, hM, Merr)
ha = array2hist(a, ha, aerr)
hc = array2hist(c, hc, cerr)

hA.GetYaxis().SetTitle('b field correction')
he.GetYaxis().SetTitle('material correction')
hM.GetYaxis().SetTitle('alignment correction')
ha.GetYaxis().SetTitle('material correction (resolution) a^2')
hc.GetYaxis().SetTitle('hit position (resolution) c^2')

hA.GetXaxis().SetTitle('#eta')
he.GetXaxis().SetTitle('#eta')
hM.GetXaxis().SetTitle('#eta')
ha.GetXaxis().SetTitle('#eta')
hc.GetXaxis().SetTitle('#eta')

#have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
scaleplotBinned = ROOT.TH1D("scaleBinned", "scale", nBins, onp.linspace(0, nBins, nBins+1))
scaleplotBinned = array2hist(scaleBinned, scaleplotBinned, scaleErrorBinned)

sigmaplotBinned = ROOT.TH1D("sigmaBinned", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
sigmaplotBinned = array2hist(sigmaBinned, sigmaplotBinned, sigmaErrorBinned)

plots = [scaleplotBinned,sigmaplotBinned]

if not fitMCtruth:
    scalejac,sigmajac = jax.jit(jax.jacfwd(scaleSigmaFromModelParVector))(xmodel.flatten(),etas, binCenters1, binCenters2, good_idx)

    scalesigmajac = np.stack((scalejac,sigmajac),axis=1)
    scalesigmajac = np.reshape(scalesigmajac, (-1,covmodel.shape[0]))
    covScaleSigmaModel = np.matmul(scalesigmajac,np.matmul(covmodel,scalesigmajac.T))
    scaleSigmaErrsModel = np.sqrt(np.diag(covScaleSigmaModel))
    scaleSigmaErrsModel = np.reshape(scaleSigmaErrsModel, (-1,2))

    print(scaleModel.shape, scaleSigmaErrsModel[:,0].shape)
    scaleplotModel = ROOT.TH1D("scaleModel", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    scaleplotModel = array2hist(scaleModel, scaleplotModel, scaleSigmaErrsModel[:,0])

    sigmaplotModel = ROOT.TH1D("sigmaModel", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
    sigmaplotModel = array2hist(sigmaModel, sigmaplotModel, scaleSigmaErrsModel[:,1])

    plots.append(scaleplotModel)
    plots.append(sigmaplotModel)

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]
        for plot in plots:
            plot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

    for plot in plots:
        plot.GetXaxis().LabelsOption("v")

filename = 'calibration'
if fitMCtruth:
    filename += 'MCtruth'
else:
    filename += 'MC'
filename += '.root'
f = ROOT.TFile(filename, 'recreate')
f.cd()

hA.Write()
he.Write()
hM.Write()
ha.Write()
hc.Write()

for plot in plots:
    plot.Write()

