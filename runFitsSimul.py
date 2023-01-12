import os
import glob
import multiprocessing
import h5py

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
from scipy.stats import chisquare
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
import itertools

import matplotlib.pyplot as plt
import mplhep as hep
# matplotlib stuff
plt.style.use([hep.style.ROOT])
# hep.cms.label(loc=0, year=2016, lumi=35.9, data=True)
# hep.cms.text('Simulation')

from fittingFunctionsBinned import scaleFromModelPars, scaleSqFromModelParsFixedMat, splitTransformPars, nllBinsFromSignalBinParsJ,nllBinsFromSignalBinParsZ, chi2LBinsSimul, scaleSqSigmaSqFromBinsPars, scaleSqSigmaSqFromBinsPars, scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars, plotsMass, kernelpdf, exppdf, nllBinsFromBinPars, nllBinsFromBinParsBW, scaleSqSigmaSqFromBinsParsBW, lorentzianpdf
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

def array2hist(array, hist, err):
    if type(hist).__cpp_name__=='TH3D':
        for i in range(1,hist.GetNbinsX()+1):
            for j in range(1,hist.GetNbinsY()+1):
                for k in range(1,hist.GetNbinsZ()+1):
                    hist.SetBinContent(i,j,k,array[i-1,j-1,k-1])
                    hist.SetBinError(i,j,k,err[i-1,j-1,k-1])
    elif type(hist).__cpp_name__=='TH2D':
        for i in range(1,hist.GetNbinsX()+1):
            for j in range(1,hist.GetNbinsY()+1):
                hist.SetBinContent(i,j,array[i-1,j-1])
                hist.SetBinError(i,j,err[i-1,j-1])
    elif type(hist).__cpp_name__=='TH1D':
        for i in range(1,hist.GetNbinsX()+1):
            hist.SetBinContent(i,array[i-1])
            hist.SetBinError(i,err[i-1])
    else:
        print(hist.GetName(),type(hist).__cpp_name__)
        print("type not recognized")
    return hist

parser = argparse.ArgumentParser('')
parser.add_argument('-isData', '--isData', default=False, action='store_true', help='data or mc reco')
parser.add_argument('-fitCalibration', '--fitCalibration', default=False, action='store_true', help='run fit of calibration parameters after scale/sigma')
parser.add_argument('-debugPlots', '--debugPlots', default=False, action='store_true', help='print a huge amount of plots for debugging purposes')
parser.add_argument('-closure', '--closure', default=False, action='store_true', help='run closure')


args = parser.parse_args()
isData = args.isData
fitCalibration = args.fitCalibration
debugPlots = args.debugPlots
closure = args.closure

closure_flag = "_corr" if closure else ""

fileJ = open("calInputJ{}_24etaBins_1ptBins{}.pkl".format("DATA" if isData else "MC",closure_flag), "rb")
fileZ = open("calInputZ{}_24etaBins_1ptBins{}.pkl".format("DATA" if isData else "MC",closure_flag), "rb")
pkgJ = pickle.load(fileJ)
datasetJ = pkgJ['dataset']
etas = pkgJ['edges'][0]
massesJ = pkgJ['edges'][-2]
if fitCalibration:
    binCenters1J = pkgJ['binCenters1']
    binCenters2J = pkgJ['binCenters2']
good_idxJ = pkgJ['good_idx']
filegenJ = open("calInputJMCgen_24etaBins_1ptBins_smeared.pkl", "rb")
datasetgenJ = pickle.load(filegenJ)
datasetgenJ = datasetgenJ[good_idxJ]
print(datasetgenJ.shape, 'gen J')
masses_genJ=massesJ
print(massesJ)
pkgZ = pickle.load(fileZ)
datasetZ = pkgZ['dataset']
massesZ = pkgZ['edges'][-2]
if fitCalibration:
    binCenters1Z = pkgZ['binCenters1']
    binCenters2Z = pkgZ['binCenters2']
    
good_idxZ = pkgZ['good_idx']

filegenZ = open("calInputZMCgen_24etaBins_1ptBins_smeared.pkl", "rb")
datasetgenZ = pickle.load(filegenZ)
datasetgenZ = datasetgenZ[good_idxZ]
print(datasetgenZ.shape, 'gen Z')
masses_genZ=massesZ
nEtaBins = len(etas)-1
nBinsJ = datasetJ.shape[0]
nBinsgenJ = datasetgenJ.shape[0]
print('J', nBinsJ, datasetJ.shape[1], datasetgenJ.shape[1])
nEtaBins = len(etas)-1
nBinsZ = datasetZ.shape[0]
nBinsgenZ = datasetgenZ.shape[0]
print('Z', nBinsZ, datasetZ.shape[1], datasetgenZ.shape[1])
scaleJ = np.ones((nBinsJ,),dtype='float64')
sigmaJ = 6e-3*np.ones((nBinsJ,),dtype='float64')
fbkgJ = np.zeros((nBinsJ,),dtype='float64')
slopeJ = np.zeros((nBinsJ,),dtype='float64')
scaleZ = 91*np.ones((nBinsZ,),dtype='float64')
sigmaZ = np.ones((nBinsZ,),dtype='float64')
fbkgZ = np.zeros((nBinsZ,),dtype='float64')
slopeZ = np.zeros((nBinsZ,),dtype='float64')

# plotsBkg(scaleZ,scaleZ,sigmaZ,fbkgZ,slopeZ,datasetZ,datasetgenZ,massesZ,masses_genZ,False,etas, good_idxZ, isData)
# assert(0)

#preliminary fit to get rid of gamma* dominated bins

nllBinspartial = functools.partial(nllBinsFromBinParsBW,masses=massesZ,masses_gen=masses_genZ)
xscale = np.stack([scaleZ,sigmaZ,fbkgZ,slopeZ],axis=-1)
xscale = np.zeros_like(xscale)
#parallel fit for scale, sigma, fbkg, slope in bins
xres = pmin(nllBinspartial, xscale, args=(datasetZ,datasetgenZ))
# xres = xscale
#compute covariance matrices of scale and sigma from binned fit
def hnll(x, datasetZ,datasetgenZ):
    #compute the hessian wrt internal fit parameters in each bin
    hess = jax.hessian(nllBinspartial)
    #invert to get the hessian
    cov = np.linalg.inv(hess(x,datasetZ,datasetgenZ))
    #compute the jacobian for scale and sigma squared wrt internal fit parameters
    jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsParsBW)(x)
    jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
    #compute covariance matrix for scalesq and sigmasq
    covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
    #invert again to get the hessian
    hscalesigmasq = np.linalg.inv(covscalesigmasq)
    return hscalesigmasq, covscalesigmasq
fh = jax.jit(jax.vmap(hnll))
hScaleSqSigmaSqBinnedZ, hCovScaleSqSigmaSqBinnedZ = fh(xres,datasetZ,datasetgenZ,)

scaleSqBinnedZ, sigmaSqBinnedZ = scaleSqSigmaSqFromBinsParsBW(xres)
scaleBinnedZ = np.sqrt(scaleSqBinnedZ)
sigmaBinnedZ = np.sqrt(sigmaSqBinnedZ)
scaleSqSigmaSqErrorsBinnedZ = np.sqrt(np.diagonal(hCovScaleSqSigmaSqBinnedZ, axis1=-1, axis2=-2))
scaleSqErrorBinnedZ = scaleSqSigmaSqErrorsBinnedZ[:,0]
sigmaSqErrorBinnedZ = scaleSqSigmaSqErrorsBinnedZ[:,1]
scaleErrorBinnedZ = 0.5*scaleSqErrorBinnedZ/scaleBinnedZ
sigmaErrorBinnedZ = 0.5*sigmaSqErrorBinnedZ/sigmaBinnedZ
print(scaleBinnedZ, '+/-', scaleErrorBinnedZ)
print(sigmaBinnedZ, '+/-', sigmaErrorBinnedZ)

scaleBinnedPatchedZ = onp.zeros((nEtaBins,nEtaBins,5,5),dtype='float64')
scaleBinnedPatchedZ[good_idxZ] = scaleBinnedZ

scaleBinnedBW = scaleBinnedZ
sigmaBinnedBW = sigmaBinnedZ
# # compute chi2
massWidthZ = massesZ[1:]-massesZ[:-1]
massWidthZ = massWidthZ[np.newaxis,:]
ndataZ = np.sum(datasetgenZ,axis=-1)
nsigZ = (1.-fbkgZ)*ndataZ
nbkgZ = fbkgZ*ndataZ
BWpdf = nsigZ[:,np.newaxis]*massWidthZ*lorentzianpdf(scaleBinnedBW,sigmaBinnedBW,massesZ)
th1 = np.digitize(88,massesZ)
th2 = np.digitize(93,massesZ)

datasetgenPatchedZ = onp.zeros((nEtaBins,nEtaBins,5,5,1000),dtype='float64')
datasetgenPatchedZ[good_idxZ] = datasetgenZ

BWpdfPatched = onp.zeros((nEtaBins,nEtaBins,5,5,1000),dtype='float64')
BWpdfPatched[good_idxZ] = BWpdf

chi2 = chisquare(datasetgenZ[...,:], BWpdf[...,:],ddof=2,axis=-1)[0]
nbinsred = datasetgenZ[...,:].shape[1]-2-1

chi2Patched = chisquare(datasetgenPatchedZ[...,:], BWpdfPatched[...,:],ddof=2,axis=-1)[0]
nbinsredPatched = datasetgenPatchedZ[...,:].shape[-1]-2-1

# load smeared Z for turtle fits
filegenZ = open("calInputZMCgen_24etaBins_1ptBins_smeared.pkl", "rb")
datasetgenZ = pickle.load(filegenZ)
datasetgenZ = datasetgenZ[good_idxZ]

print(datasetgenZ[...,:].shape, BWpdf[...,:].shape,chi2.shape)
good_idxZBW = np.nonzero((scaleBinnedZ>89) & (scaleBinnedZ<92) & (chi2/nbinsred<2.))
good_idxZ = np.nonzero((scaleBinnedPatchedZ>89) & (scaleBinnedPatchedZ<92) & (chi2Patched/nbinsredPatched<2.))

print(len(good_idxZBW))
print(len(good_idxZ))

datasetgenZ = datasetgenZ[good_idxZBW]
datasetZ = datasetZ[good_idxZBW]
nBinsZ = datasetZ.shape[0]
nBinsgenZ = datasetgenZ.shape[0]
if fitCalibration:
    binCenters1Z = binCenters1Z[good_idxZBW]
    binCenters2Z = binCenters2Z[good_idxZBW]

print(nBinsZ, datasetZ.shape[1], datasetgenZ.shape[1])

scaleZ = np.ones((nBinsZ,),dtype='float64')
sigmaZ = 6e-3*np.ones((nBinsZ,),dtype='float64')
# sigmaZ =1*np.ones((nBinsZ,),dtype='float64')
fbkgZ = np.zeros((nBinsZ,),dtype='float64')
slopeZ = np.zeros((nBinsZ,),dtype='float64')

if not isData:
    print("start fitting J")
    # xscale = np.stack([scaleJ,sigmaJ],axis=-1)
    # xscale = np.zeros_like(xscale)
    # nllBinspartialJ = functools.partial(nllBinsFromSignalBinParsJ,masses=massesJ, masses_gen=masses_genJ, isJ=True)
    # #parallel fit for scale, sigma, fbkg, slope in bins
    # xresJ = pmin(nllBinspartialJ, xscale, args=(fbkgJ, slopeJ, datasetJ,datasetgenJ))
    # #compute covariance matrices of scale and sigma from binned fit
    # def hnll(x, fbkgJ, slopeJ, datasetJ,datasetgenJ):
    #     #compute the hessian wrt internal fit parameters in each bin
    #     hess = jax.hessian(nllBinspartialJ)
    #     #invert to get the hessian
    #     cov = np.linalg.inv(hess(x,fbkgJ, slopeJ,datasetJ,datasetgenJ,))
    #     #compute the jacobian for scale and sigma squared wrt internal fit parameters
    #     jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
    #     jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
    #     #compute covariance matrix for scalesq and sigmasq
    #     covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
    #     #invert again to get the hessian
    #     hscalesigmasq = np.linalg.inv(covscalesigmasq)
    #     return hscalesigmasq, covscalesigmasq
    # fh = jax.jit(jax.vmap(hnll))
    # hScaleSqSigmaSqBinnedJ, hCovScaleSqSigmaSqBinnedJ = fh(xresJ,fbkgJ, slopeJ,datasetJ,datasetgenJ,)
    
    nllBinspartialJ = functools.partial(nllBinsFromBinPars,masses=massesJ,masses_gen=masses_genJ, isJ=True)
    xscale = np.stack([scaleJ,sigmaJ,fbkgJ,slopeJ],axis=-1)
    xscale = np.zeros_like(xscale)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xresJ = pmin(nllBinspartialJ, xscale, args=(datasetJ,datasetgenJ))
    #xres = xscal
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, datasetJ,datasetgenJ):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartialJ)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,datasetJ,datasetgenJ,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinnedJ, hCovScaleSqSigmaSqBinnedJ = fh(xresJ,datasetJ,datasetgenJ,)
    print("end fitting J")
    xscale = np.stack([scaleZ,sigmaZ],axis=-1)
    xscale = np.zeros_like(xscale)
    nllBinspartialZ = functools.partial(nllBinsFromSignalBinParsZ,masses=massesZ, masses_gen=masses_genZ, isJ=False)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xresZ = pmin(nllBinspartialZ, xscale, args=(fbkgZ, slopeZ, datasetZ,datasetgenZ))
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, fbkgZ, slopeZ, datasetZ,datasetgenZ):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartialZ)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,fbkgZ, slopeZ,datasetZ,datasetgenZ,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinnedZ, hCovScaleSqSigmaSqBinnedZ = fh(xresZ,fbkgZ, slopeZ,datasetZ,datasetgenZ,)
else: 
    nllBinspartialJ = functools.partial(nllBinsFromBinPars,masses=massesJ,masses_gen=masses_genJ, isJ=True)
    xscale = np.stack([scaleJ,sigmaJ,fbkgJ,slopeJ],axis=-1)
    xscale = np.zeros_like(xscale)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xresJ = pmin(nllBinspartialJ, xscale, args=(datasetJ,datasetgenJ))
    #xres = xscal
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, datasetJ,datasetgenJ):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartialJ)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,datasetJ,datasetgenJ,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinnedJ, hCovScaleSqSigmaSqBinnedJ = fh(xresJ,datasetJ,datasetgenJ,)

    # nllBinspartialZ = functools.partial(nllBinsFromBinPars,masses=massesZ,masses_gen=masses_genZ)
    # xscale = np.stack([scaleZ,sigmaZ,fbkgZ,slopeZ],axis=-1)
    # xscale = np.zeros_like(xscale)
    # #parallel fit for scale, sigma, fbkg, slope in bins
    # xresZ = pmin(nllBinspartialZ, xscale, args=(datasetZ,datasetgenZ))
    # #xres = xscal
    # #compute covariance matrices of scale and sigma from binned fit
    # def hnll(x, datasetZ,datasetgenZ):
    #     #compute the hessian wrt internal fit parameters in each bin
    #     hess = jax.hessian(nllBinspartialZ)
    #     #invert to get the hessian
    #     cov = np.linalg.inv(hess(x,datasetZ,datasetgenZ,))
    #     #compute the jacobian for scale and sigma squared wrt internal fit parameters
    #     jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
    #     jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
    #     #compute covariance matrix for scalesq and sigmasq
    #     covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
    #     #invert again to get the hessian
    #     hscalesigmasq = np.linalg.inv(covscalesigmasq)
    #     return hscalesigmasq, covscalesigmasq
    # fh = jax.jit(jax.vmap(hnll))
    # hScaleSqSigmaSqBinnedZ, hCovScaleSqSigmaSqBinnedZ = fh(xresZ,datasetZ,datasetgenZ,)

    #fit Z without bkg
    xscale = np.stack([scaleZ,sigmaZ],axis=-1)
    xscale = np.zeros_like(xscale)
    nllBinspartialZ = functools.partial(nllBinsFromSignalBinPars,masses=massesZ, masses_gen=masses_genZ)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xresZ = pmin(nllBinspartialZ, xscale, args=(fbkgZ, slopeZ, datasetZ,datasetgenZ))
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, fbkgZ, slopeZ, datasetZ,datasetgenZ):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartialZ)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,fbkgZ, slopeZ,datasetZ,datasetgenZ,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinnedZ, hCovScaleSqSigmaSqBinnedZ = fh(xresZ,fbkgZ, slopeZ,datasetZ,datasetgenZ,)

if isData:
    fbkgJ, slopeJ = bkgModelFromBinPars(xresJ)
    fbkgZ, slopeZ = bkgModelFromBinPars(xresZ)
else:
    fbkgJ = np.zeros((nBinsJ,),dtype='float64')
    slopeJ = np.zeros((nBinsJ,),dtype='float64')
    fbkgZ = np.zeros((nBinsZ,),dtype='float64')
    slopeZ = np.zeros((nBinsZ,),dtype='float64')

scaleSqBinnedJ, sigmaSqBinnedJ = scaleSqSigmaSqFromBinsPars(xresJ)
scaleBinnedJ = np.sqrt(scaleSqBinnedJ)
sigmaBinnedJ = np.sqrt(sigmaSqBinnedJ)
scaleSqSigmaSqErrorsBinnedJ = np.sqrt(np.diagonal(hCovScaleSqSigmaSqBinnedJ, axis1=-1, axis2=-2))
scaleSqErrorBinnedJ = scaleSqSigmaSqErrorsBinnedJ[:,0]
print('scaleSqErrorBinnedJ',scaleSqErrorBinnedJ)
sigmaSqErrorBinnedJ = scaleSqSigmaSqErrorsBinnedJ[:,1]
scaleErrorBinnedJ = 0.5*scaleSqErrorBinnedJ/scaleBinnedJ
sigmaErrorBinnedJ = 0.5*sigmaSqErrorBinnedJ/sigmaBinnedJ
print(scaleBinnedJ, '+/-', scaleErrorBinnedJ)
print(sigmaBinnedJ, '+/-', sigmaErrorBinnedJ)

scaleSqBinnedZ, sigmaSqBinnedZ = scaleSqSigmaSqFromBinsPars(xresZ)
scaleBinnedZ = np.sqrt(scaleSqBinnedZ)
sigmaBinnedZ = np.sqrt(sigmaSqBinnedZ)
scaleSqSigmaSqErrorsBinnedZ = np.sqrt(np.diagonal(hCovScaleSqSigmaSqBinnedZ, axis1=-1, axis2=-2))
scaleSqErrorBinnedZ = scaleSqSigmaSqErrorsBinnedZ[:,0]
sigmaSqErrorBinnedZ = scaleSqSigmaSqErrorsBinnedZ[:,1]
scaleErrorBinnedZ = 0.5*scaleSqErrorBinnedZ/scaleBinnedZ
sigmaErrorBinnedZ = 0.5*sigmaSqErrorBinnedZ/sigmaBinnedZ
print(scaleBinnedZ, '+/-', scaleErrorBinnedZ)
print(sigmaBinnedZ, '+/-', sigmaErrorBinnedZ)

fileOut = "fitsJ{}{}.hdf5".format("DATA" if isData else "MC",closure_flag)

with h5py.File(fileOut, mode="w") as f:
    dtype = 'float64'
    dset_scale = f.create_dataset('scale', scaleBinnedJ.shape, dtype=dtype)
    dset_scale[...] = scaleBinnedJ
    dset_scale = f.create_dataset('scaleErr', scaleErrorBinnedJ.shape, dtype=dtype)
    dset_scale[...] = scaleErrorBinnedJ
    dset_sigma = f.create_dataset('sigma', sigmaBinnedJ.shape, dtype=dtype)
    dset_sigma[...] = sigmaBinnedJ
    dset_sigma = f.create_dataset('sigmaErr', sigmaErrorBinnedJ.shape, dtype=dtype)
    dset_sigma[...] = sigmaErrorBinnedJ
    dset_scale = f.create_dataset('scaleSq', scaleSqBinnedJ.shape, dtype=dtype)
    dset_scale[...] = scaleSqBinnedJ
    dset_scale = f.create_dataset('scaleSqErr', scaleSqErrorBinnedJ.shape, dtype=dtype)
    dset_scale[...] = scaleSqErrorBinnedJ
    dset_sigma = f.create_dataset('sigmaSq', sigmaSqBinnedJ.shape, dtype=dtype)
    dset_sigma[...] = sigmaSqBinnedJ
    dset_sigma = f.create_dataset('sigmaSqErr', sigmaSqErrorBinnedJ.shape, dtype=dtype)
    dset_sigma[...] = sigmaSqErrorBinnedJ
    dset_good_idx = f.create_dataset('good_idx', np.asarray(good_idxJ).shape, dtype="int")
    dset_good_idx[...] = np.asarray(good_idxJ)
    dset_etas = f.create_dataset('etas', etas.shape, dtype=dtype)
    dset_etas[...] = etas

fileOut = "fitsZ{}{}.hdf5".format("DATA" if isData else "MC",closure_flag)

with h5py.File(fileOut, mode="w") as f:
    dtype = 'float64'
    dset_scale = f.create_dataset('scale', scaleBinnedZ.shape, dtype=dtype)
    dset_scale[...] = scaleBinnedZ
    dset_scale = f.create_dataset('scaleErr', scaleErrorBinnedZ.shape, dtype=dtype)
    dset_scale[...] = scaleErrorBinnedZ
    dset_sigma = f.create_dataset('sigma', sigmaBinnedZ.shape, dtype=dtype)
    dset_sigma[...] = sigmaBinnedZ
    dset_scale = f.create_dataset('scaleSq', scaleSqBinnedZ.shape, dtype=dtype)
    dset_scale[...] = scaleSqBinnedZ
    dset_scale = f.create_dataset('scaleSqErr', scaleSqErrorBinnedZ.shape, dtype=dtype)
    dset_scale[...] = scaleSqErrorBinnedZ
    dset_sigma = f.create_dataset('sigmaErr', sigmaErrorBinnedZ.shape, dtype=dtype)
    dset_sigma[...] = sigmaErrorBinnedZ
    dset_sigma = f.create_dataset('sigmaSq', sigmaSqBinnedZ.shape, dtype=dtype)
    dset_sigma[...] = sigmaSqBinnedZ
    dset_sigma = f.create_dataset('sigmaSqErr', sigmaSqErrorBinnedZ.shape, dtype=dtype)
    dset_sigma[...] = sigmaSqErrorBinnedZ
    dset_good_idx = f.create_dataset('good_idx', np.asarray(good_idxZ).shape, dtype="int")
    dset_good_idx[...] = np.asarray(good_idxZ)
    dset_etas = f.create_dataset('etas', etas.shape, dtype=dtype)
    dset_etas[...] = etas

#     # rebuild pdf
#     massWidth = masses[1:]-masses[:-1]
#     massWidth = massWidth[np.newaxis,:]
# #     ndata = np.sum(dataset,axis=-1)
# #     nsig = (1.-fbkg)*ndata
# #     nbkg = fbkg*ndata
#     sigpdf = nsig[:,np.newaxis]*massWidth*kernelpdf(scaleBinned,sigmaBinned, datasetgen, masses,masses_gen)
#     bkgpdf = nbkg[:,np.newaxis]*massWidth*exppdf(slope,masses)
#     dset_sigpdf = f.create_dataset('sigpdf', sigpdf.shape, dtype=dtype)
#     dset_sigpdf[...] = sigpdf
#     dset_bkgpdf = f.create_dataset('bkgpdf', bkgpdf.shape, dtype=dtype)
#     dset_bkgpdf[...] = bkgpdf

# if debugPlots:
#     if fitMCtruth:
#         plotsSingleMu(scaleBinned,sigmaBinned,dataset,masses)
#     else:
#         plotsBkg(scaleBinned,scaleErrorBinned,sigmaBinned,fbkg,slope,dataset,datasetgen,masses,masses_gen,isJ,etas, good_idx, isData)

#have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
scaleplotBinnedJ = ROOT.TH1D("scaleBinnedJ", "scaleJ", nBinsJ, onp.linspace(0, nBinsJ, nBinsJ+1))
scaleplotBinnedJ = array2hist(scaleBinnedJ, scaleplotBinnedJ, scaleErrorBinnedJ)

sigmaplotBinnedJ = ROOT.TH1D("sigmaBinnedJ", "sigmaJ", nBinsJ, onp.linspace(0, nBinsJ, nBinsJ+1))
sigmaplotBinnedJ = array2hist(sigmaBinnedJ, sigmaplotBinnedJ, sigmaErrorBinnedJ)

scaleplotBinnedZ = ROOT.TH1D("scaleBinnedZ", "scaleZ", nBinsZ, onp.linspace(0, nBinsZ, nBinsZ+1))
scaleplotBinnedZ = array2hist(scaleBinnedZ, scaleplotBinnedZ, scaleErrorBinnedZ)

sigmaplotBinnedZ = ROOT.TH1D("sigmaBinnedZ", "sigmaZ", nBinsZ, onp.linspace(0, nBinsZ, nBinsZ+1))
sigmaplotBinnedZ = array2hist(sigmaBinnedZ, sigmaplotBinnedZ, sigmaErrorBinnedZ)

plots = [scaleplotBinnedJ,sigmaplotBinnedJ,scaleplotBinnedZ,sigmaplotBinnedZ]

###### begin parameters fit
if fitCalibration:

    nModelParms = 6

    A = np.zeros((nEtaBins),dtype=np.float64)
    e = np.zeros((nEtaBins),dtype=np.float64)
    M = np.zeros((nEtaBins),dtype=np.float64)
    a = 1e-6*np.ones((nEtaBins),dtype=np.float64)
    c = 10e-9*np.ones((nEtaBins),dtype=np.float64)
    b = np.zeros((nEtaBins),dtype=np.float64)
    d = 3.7*np.ones((nEtaBins),dtype=np.float64)

    xmodel = np.stack((A,e,M,a,c,b),axis=-1)

    chi2 = chi2LBinsSimul(xmodel, scaleSqBinnedJ, sigmaSqBinnedJ, hScaleSqSigmaSqBinnedJ, etas,binCenters1J, binCenters2J, good_idxJ, scaleSqBinnedZ, sigmaSqBinnedZ, hScaleSqSigmaSqBinnedZ,binCenters1Z, binCenters2Z, good_idxZ)

    print(chi2)

    xmodel = pmin(chi2LBinsSimul, xmodel.flatten(), args=(scaleSqBinnedJ, sigmaSqBinnedJ, hScaleSqSigmaSqBinnedJ, etas,binCenters1J, binCenters2J, good_idxJ, scaleSqBinnedZ, sigmaSqBinnedZ, hScaleSqSigmaSqBinnedZ,binCenters1Z, binCenters2Z, good_idxZ), doParallel=False)
    xmodel = xmodel.reshape((-1,nModelParms))

    fgchi2 = jax.jit(jax.value_and_grad(chi2LBinsSimul))
    hchi2 = jax.jit(jax.hessian(chi2LBinsSimul))

    chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqBinnedJ, sigmaSqBinnedJ, hScaleSqSigmaSqBinnedJ, etas,binCenters1J, binCenters2J, good_idxJ, scaleSqBinnedZ, sigmaSqBinnedZ, hScaleSqSigmaSqBinnedZ,binCenters1Z, binCenters2Z, good_idxZ)
    chi2hess = hchi2(xmodel.flatten(), scaleSqBinnedJ, sigmaSqBinnedJ, hScaleSqSigmaSqBinnedJ, etas,binCenters1J, binCenters2J, good_idxJ, scaleSqBinnedZ, sigmaSqBinnedZ, hScaleSqSigmaSqBinnedZ,binCenters1Z, binCenters2Z, good_idxZ)

    hmodel = chi2hess
    covmodel = np.linalg.inv(chi2hess)
    invhess = covmodel

    valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqBinnedJ, sigmaSqBinnedJ, hScaleSqSigmaSqBinnedJ, etas,binCenters1J, binCenters2J, good_idxJ, scaleSqBinnedZ, sigmaSqBinnedZ, hScaleSqSigmaSqBinnedZ,binCenters1Z, binCenters2Z, good_idxZ)

    ndof = 2*(nBinsJ+nBinsZ) - nEtaBins*nModelParms
    edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

    print("nEtaBins", nEtaBins)
    print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

    errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

    A,e,M,a,b, c = modelParsFromParVector(xmodel)

    scaleSqModelJ = scaleSqFromModelPars(A, e, M, etas, binCenters1J, binCenters2J, good_idxJ)
    sigmaSqModelJ = sigmaSqFromModelPars(a, b, c, etas, binCenters1J, binCenters2J, good_idxJ)

    scaleModelJ = np.sqrt(scaleSqModelJ)
    sigmaModelJ = np.sqrt(sigmaSqModelJ)

    scaleSqModelZ = scaleSqFromModelParsFixedMat(A, M, etas, binCenters1Z, binCenters2Z, good_idxZ)
    # scaleSqModelZ = scaleSqFromModelPars(A, e, M, etas, binCenters1Z, binCenters2Z, good_idxZ)
    sigmaSqModelZ = sigmaSqFromModelPars(a, b, c, etas, binCenters1Z, binCenters2Z, good_idxZ)

    scaleModelZ = np.sqrt(scaleSqModelZ)
    sigmaModelZ = np.sqrt(sigmaSqModelZ)

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
    plt.savefig("corrmatrix{}.pdf".format("Data" if isData else "MC"))

    print("computing scales and errors:")

    # ndata = np.sum(dataset,axis=-1)

    Aerr = errsmodel[:,0]
    eerr = errsmodel[:,1]
    Merr = errsmodel[:,2]
    aerr = errsmodel[:,3]
    cerr = errsmodel[:,4]
    berr = errsmodel[:,5]
    derr = errsmodel[:,6]

    etaarr = onp.array(etas.tolist())
    hA = ROOT.TH1D("A", "A", nEtaBins, etaarr)
    he = ROOT.TH1D("e", "e", nEtaBins, etaarr)
    hM = ROOT.TH1D("M", "M", nEtaBins, etaarr)
    ha = ROOT.TH1D("a", "a", nEtaBins, etaarr)
    hc = ROOT.TH1D("c", "c", nEtaBins, etaarr)
    hb = ROOT.TH1D("b", "b", nEtaBins, etaarr)
    hd = ROOT.TH1D("d", "d", nEtaBins, etaarr)

    hA = array2hist(A, hA, Aerr)
    he = array2hist(e, he, eerr)
    hM = array2hist(M, hM, Merr)
    ha = array2hist(a, ha, aerr)
    hc = array2hist(c, hc, cerr)
    hb = array2hist(b, hb, berr)
    hd = array2hist(d, hd, derr)

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

    # scalejacJ,sigmajacJ = jax.jit(jax.jacfwd(scaleSigmaFromModelParVector))(xmodel.flatten(),etas, binCenters1J, binCenters2J, good_idxJ)

    # scalesigmajacJ = np.stack((scalejacJ,sigmajacJ),axis=1)
    # scalesigmajacJ = np.reshape(scalesigmajacJ, (-1,covmodel.shape[0]))
    # covScaleSigmaModel = np.matmul(scalesigmajac,np.matmul(covmodel,scalesigmajac.T))
    # scaleSigmaErrsModel = np.sqrt(np.diag(covScaleSigmaModel))
    # scaleSigmaErrsModel = np.reshape(scaleSigmaErrsModel, (-1,2))

    # print(scaleModel.shape, scaleSigmaErrsModel[:,0].shape)
    # scaleplotModel = ROOT.TH1D("scaleModel", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    # scaleplotModel = array2hist(scaleModel, scaleplotModel, scaleSigmaErrsModel[:,0])

    # sigmaplotModel = ROOT.TH1D("sigmaModel", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
    # sigmaplotModel = array2hist(sigmaModel, sigmaplotModel, scaleSigmaErrsModel[:,1])

    # plots.append(scaleplotModel)
    # plots.append(sigmaplotModel)


if fitCalibration:
    if not isData:
        f = ROOT.TFile("calibrationMC{}.root".format(closure_flag), 'recreate')
    else:
        f = ROOT.TFile("calibrationDATA{}.root".format(closure_flag), 'recreate')
    f.cd()

    hA.Write()
    he.Write()
    hM.Write()
    ha.Write()
    hc.Write()
    hb.Write()
    hd.Write()

    correlationHist = ROOT.TH2D('correlation_matrix', 'correlation matrix', 24*nModelParms, 0., 1., 24*nModelParms, 0., 1.)
    covarianceHist  = ROOT.TH2D('covariance_matrix', 'covariance matrix', 24*nModelParms, 0., 1., 24*nModelParms, 0., 1.)
    correlationHist.GetZaxis().SetRangeUser(-1., 1.)

    array2hist(corr, correlationHist)
    array2hist(invhess, covarianceHist)

    correlationHist.Write()
    covarianceHist.Write()

    for plot in plots:
        plot.Write()

# scale and sigma plots in pdf
isJ = False

bins=np.linspace(1, scaleBinnedZ.shape[0]+1, scaleBinnedZ.shape[0]+1)
binsC = (bins[:-1] + bins[1:]) / 2
# fig, ax1 = plt.subplots()
# ax1.set_title("scale_{}".format("data" if isData else "mc"), fontsize=18)
# hep.histplot(scaleBinnedJ,bins,yerr=scaleErrorBinnedJ,histtype = 'errorbar', ax=ax1, label = ["binned scale fit"])

# if fitCalibration:
#     ax1.plot(binsC, scaleModel, color="red", label = "fit to model", zorder=10)
#     ax1.fill_between(binsC, scaleModel-scaleSigmaErrsModel[:,0], scaleModel+scaleSigmaErrsModel[:,0], color="red", alpha=0.4)

# ax1.set_ylim([np.min(scaleBinned)-0.0005, np.max(scaleBinned)+0.0005])
# ax1.legend(loc='upper right', frameon=True)
# ax1.set_xticklabels([])
# plt.tight_layout()
# plt.savefig("scale_{}_{}{}.pdf".format("J" if isJ else "Z", "data" if isData else "MC",closure_flag))
# plt.clf()

fig, ax1 = plt.subplots()
ax1.set_title("sigma_{}".format("data" if isData else "mc"), fontsize=18)
hep.histplot(sigmaBinnedZ,bins,yerr=sigmaErrorBinnedZ,histtype = 'errorbar', ax=ax1, label = ["binned sigma fit"])

# if fitCalibration:
#     ax1.plot(binsC, sigmaModelZ, color="red", label = "fit to model", zorder=10)
#     ax1.fill_between(binsC, sigmaModelZ-scaleSigmaErrsModelZ[:,1], sigmaModelZ+scaleSigmaErrsModelZ[:,1], color="red", alpha=0.4)

ax1.set_ylim([np.min(sigmaBinnedZ)-0.001, np.max(sigmaBinnedZ)+0.001])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig("sigma_{}_{}_simul.png".format("J" if isJ else "Z", "data" if isData else "MC"))
plt.clf()

isJ = True

bins=np.linspace(1, scaleBinnedJ.shape[0]+1, scaleBinnedJ.shape[0]+1)
binsC = (bins[:-1] + bins[1:]) / 2

fig, ax1 = plt.subplots()
ax1.set_title("scale_{}".format("data" if isData else "mc"), fontsize=18)
hep.histplot(scaleBinnedJ,bins,yerr=scaleErrorBinnedJ,histtype = 'errorbar', ax=ax1, label = ["binned scale fit"])

# if fitCalibration:
#     ax1.plot(binsC, scaleModel, color="red", label = "fit to model", zorder=10)
#     ax1.fill_between(binsC, scaleModel-scaleSigmaErrsModel[:,0], scaleModel+scaleSigmaErrsModel[:,0], color="red", alpha=0.4)

# ax1.set_ylim([np.min(scaleBinned)-0.0005, np.max(scaleBinned)+0.0005])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig("scale_{}_{}.pdf".format("J" if isJ else "Z", "data" if isData else "MC"))
plt.clf()

fig, ax1 = plt.subplots()
ax1.set_title("sigma_{}".format("data" if isData else "mc"), fontsize=18)
hep.histplot(sigmaBinnedJ,bins,yerr=sigmaErrorBinnedJ,histtype = 'errorbar', ax=ax1, label = ["binned sigma fit"])

# if fitCalibration:
#     ax1.plot(binsC, sigmaModelZ, color="red", label = "fit to model", zorder=10)
#     ax1.fill_between(binsC, sigmaModelZ-scaleSigmaErrsModelZ[:,1], sigmaModelZ+scaleSigmaErrsModelZ[:,1], color="red", alpha=0.4)

ax1.set_ylim([np.min(sigmaBinnedJ)-0.001, np.max(sigmaBinnedJ)+0.001])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig("sigma_{}_{}_simul.png".format("J" if isJ else "Z", "data" if isData else "MC"))
plt.clf()

# files = glob.glob('/PlotsSigma/*')
# for f in files:
#     os.remove(f)

# # separate plots per eta bins
# full = onp.zeros((nEtaBins,nEtaBins,6,6),dtype='float64')
# full[good_idxJ] = sigmaBinnedJ
# sigmaBinnedPatchedJ = full

# full = onp.zeros((nEtaBins,nEtaBins,6,6),dtype='float64')
# full[good_idxJ] = sigmaErrorBinnedJ
# sigmaErrorBinnedPatchedJ = full

# full = onp.zeros((nEtaBins,nEtaBins,5,5),dtype='float64')
# full[good_idxZ] = sigmaBinnedZ
# sigmaBinnedPatchedZ = full

# full = onp.zeros((nEtaBins,nEtaBins,5,5),dtype='float64')
# full[good_idxZ] = sigmaErrorBinnedZ
# sigmaErrorBinnedPatchedZ = full

# etasC = (etas[:-1] + etas[1:]) / 2
# ptsZ = np.square(np.array([8.,25.,38, 44, 48.7, 100.]))
# ptsJ = np.square(np.array([1.1, 2.560737, 4.02147399, 5.49946864, 7.02340517, 8.54734171, 25.]))
# ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
# ptsJC = (ptsJ[:-1] + ptsJ[1:]) / 2
# pts=np.concatenate((ptsJ,ptsZ))
# ptsC = (pts[:-1] + pts[1:]) / 2

# for ieta1 in range(len(etasC)):
#     for ieta2 in range(len(etasC)):
#         if not (round(etasC[ieta1],2)==-0.3 and round(etasC[ieta2],2)==-0.1): continue
#         if np.all(sigmaBinnedPatchedJ[ieta1,ieta2,:,:]==0) and np.all(sigmaBinnedPatchedZ[ieta1,ieta2,:,:]==0):
#             continue
#         sigmasJ = []
#         sigmasErrJ = []
#         sigmasZ = []
#         sigmasErrZ = []
#         for ipt1J in range(len(ptsJC)):
#             try:
#                 sigmasJ.append(np.average(sigmaBinnedPatchedJ[ieta1,ieta2,ipt1J,...][sigmaBinnedPatchedJ[ieta1,ieta2,ipt1J,...]!=0],weights=np.reciprocal(np.square(sigmaErrorBinnedPatchedJ[ieta1,ieta2,ipt1J,...][sigmaBinnedPatchedJ[ieta1,ieta2,ipt1J,...]!=0]))))
#             except ZeroDivisionError:
#                 sigmasJ.append(0)
#             try:
#                 sigmasErrJ.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmaErrorBinnedPatchedJ[ieta1,ieta2,ipt1J,...][sigmaBinnedPatchedJ[ieta1,ieta2,ipt1J,...]!=0])))))
#             except ZeroDivisionError:
#                 sigmasErrJ.append(0)
        
#         for ipt1Z in range(len(ptsZC)):
#             print(round(ptsZC[ipt1Z],2), sigmaBinnedPatchedZ[ieta1,ieta2,ipt1Z,...])
#             try:
#                 sigmasZ.append(np.average(sigmaBinnedPatchedZ[ieta1,ieta2,ipt1Z,...][sigmaBinnedPatchedZ[ieta1,ieta2,ipt1Z,...]!=0],weights=np.reciprocal(np.square(sigmaErrorBinnedPatchedZ[ieta1,ieta2,ipt1Z,...][sigmaBinnedPatchedZ[ieta1,ieta2,ipt1Z,...]!=0]))))
#             except ZeroDivisionError:
#                 sigmasZ.append(0)
#             try:
#                 sigmasErrZ.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmaErrorBinnedPatchedZ[ieta1,ieta2,ipt1Z,...][sigmaBinnedPatchedZ[ieta1,ieta2,ipt1Z,...]!=0])))))
#             except ZeroDivisionError:
#                 sigmasErrZ.append(0)

#         sigmasJ = np.array(sigmasJ)
#         sigmasErrJ = np.array(sigmasErrJ)
#         sigmasZ = np.array(sigmasZ)
#         sigmasErrZ = np.array(sigmasErrZ)
#         print(sigmasZ, sigmasErrZ)
        
#         fig, ax1 = plt.subplots()
#         ax1.set_title("sigma {} {}".format(round(etasC[ieta1],2),round(etasC[ieta2],2)), fontsize=18)
#         ax1.errorbar(ptsJC,sigmasJ.ravel(),yerr=sigmasErrJ.ravel(),marker="v",fmt='v', label = "binned sigma fit J")
#         ax1.errorbar(ptsZC,sigmasZ.ravel(),yerr=sigmasErrZ.ravel(),marker="v",fmt='v', label = "binned sigma fit Z")
#         ax1.legend(loc='upper right', frameon=True)
#         # ax1.set_xticklabels([])
#         plt.tight_layout()
#         plt.savefig("PlotsSigma/sigma_{}_{}.png".format(round(etasC[ieta1],2),round(etasC[ieta2],2)))
#         plt.clf()
#         # assert(0)

