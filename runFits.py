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

from fittingFunctionsBinned import scaleFromModelPars, splitTransformPars, nllBinsFromBinPars, chi2LBins, scaleSqSigmaSqFromBinsPars,scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars, nllBinsFromBinParsRes, plotsSingleMu, scaleSqFromModelParsSingleMu, sigmaSqFromModelParsSingleMu, nllBinsFromSignalBinPars, plotsMass, kernelpdf, exppdf,scaleSqFromModelParsFixedMat
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
parser.add_argument('-isData', '--isData', default=False, action='store_true', help='data or mc reco')
parser.add_argument('-fitCalibration', '--fitCalibration', default=False, action='store_true', help='run fit of calibration parameters after scale/sigma')
parser.add_argument('-debugPlots', '--debugPlots', default=False, action='store_true', help='print a huge amount of plots for debugging purposes')
parser.add_argument('-closure', '--closure', default=False, action='store_true', help='run closure')


args = parser.parse_args()
isJ = args.isJ
isData = args.isData
fitCalibration = args.fitCalibration
debugPlots = args.debugPlots
closure = args.closure

closure_flag = "_corr" if closure else ""

fileJ = open("calInput{}{}_48etaBins_6ptBins{}.pkl".format('J' if isJ else 'Z',"DATA" if isData else "MC",closure_flag), "rb")
pkgJ = pickle.load(fileJ)
datasetJ = pkgJ['dataset']
# fileJ2 = open("calInput{}{}_48etaBins_6ptBins{}sm2.pkl".format('J' if isJ else 'Z',"DATA" if isData else "MC",closure_flag), "rb")
# pkgJ2 = pickle.load(fileJ2)
# datasetJ2 = pkgJ2['dataset']
# datasetJ = pkgJ['dataset']+pkgJ2['dataset']
etas = pkgJ['edges'][0]
ptsJ = pkgJ['edges'][2]
massesJ = pkgJ['edges'][-1]
if fitCalibration:
    binCenters1J = pkgJ['binCenters1']
    binCenters2J = pkgJ['binCenters2']
good_idxJ = pkgJ['good_idx']
dataset = datasetJ
pts = ptsJ
masses = massesJ
if fitCalibration:
    binCenters1 = binCenters1J
    binCenters2 = binCenters2J
good_idx= good_idxJ
filegen = open("calInput{}MCgen_48etaBins_6ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetgen = pickle.load(filegen)
datasetgen = datasetgen[good_idx]
print(datasetgen.shape, 'gen')
masses_gen=masses
nEtaBins = len(etas)-1
print(dataset.shape)
nBins = dataset.shape[0]
nBinsgen = datasetgen.shape[0]
print(nBins, dataset.shape[1], datasetgen.shape[1])
scale = np.ones((nBins,),dtype='float64')
sigma = 6e-3*np.ones((nBins,),dtype='float64')
fbkg = np.zeros((nBins,),dtype='float64')
slope = np.zeros((nBins,),dtype='float64')

# plotsBkg(scale,scale,sigma,fbkg,slope,dataset,datasetgen,masses,masses_gen,isJ,etas, good_idx, isData)
# assert(0)

if not isData or not isJ: 
    xscale = np.stack([scale,sigma],axis=-1)
    xscale = np.zeros_like(xscale)
    nllBinspartial = functools.partial(nllBinsFromSignalBinPars,masses=masses, masses_gen=masses_gen)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xres = pmin(nllBinspartial, xscale, args=(fbkg, slope, dataset,datasetgen))
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, fbkg, slope, dataset,datasetgen):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartial)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,fbkg, slope,dataset,datasetgen,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinned, hCovScaleSqSigmaSqBinned = fh(xres,fbkg, slope,dataset,datasetgen,)
else: 
    nllBinspartial = functools.partial(nllBinsFromBinPars,masses=masses,masses_gen=masses_gen)
    xscale = np.stack([scale,sigma,fbkg,slope],axis=-1)
    xscale = np.zeros_like(xscale)
    #parallel fit for scale, sigma, fbkg, slope in bins
    xres = pmin(nllBinspartial, xscale, args=(dataset,datasetgen))
    #xres = xscal
    #compute covariance matrices of scale and sigma from binned fit
    def hnll(x, dataset,datasetgen):
        #compute the hessian wrt internal fit parameters in each bin
        hess = jax.hessian(nllBinspartial)
        #invert to get the hessian
        cov = np.linalg.inv(hess(x,dataset,datasetgen,))
        #compute the jacobian for scale and sigma squared wrt internal fit parameters
        jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
        jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
        #compute covariance matrix for scalesq and sigmasq
        covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
        #invert again to get the hessian
        hscalesigmasq = np.linalg.inv(covscalesigmasq)
        return hscalesigmasq, covscalesigmasq
    fh = jax.jit(jax.vmap(hnll))
    hScaleSqSigmaSqBinned, hCovScaleSqSigmaSqBinned = fh(xres,dataset,datasetgen,)

if isData and isJ:
    fbkg, slope = bkgModelFromBinPars(xres)
else:
    fbkg = np.zeros((nBins,),dtype='float64')
    slope = np.zeros((nBins,),dtype='float64')

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

fileOut = "fits{}{}{}.hdf5".format("J" if isJ else "Z","DATA" if isData else "MC",closure_flag)

with h5py.File(fileOut, mode="w") as f:
    dtype = 'float64'
    dset_scale = f.create_dataset('scale', scaleBinned.shape, dtype=dtype)
    dset_scale[...] = scaleBinned
    dset_scale = f.create_dataset('scaleErr', scaleErrorBinned.shape, dtype=dtype)
    dset_scale[...] = scaleErrorBinned
    dset_sigma = f.create_dataset('sigma', sigmaBinned.shape, dtype=dtype)
    dset_sigma[...] = sigmaBinned
    dset_sigma = f.create_dataset('sigmaErr', sigmaErrorBinned.shape, dtype=dtype)
    dset_sigma[...] = sigmaErrorBinned
    dset_fbkg = f.create_dataset('fbkg', fbkg.shape, dtype=dtype)
    dset_fbkg[...] = fbkg
    dset_slope = f.create_dataset('slope', slope.shape, dtype=dtype)
    dset_slope[...] = slope
    dset_good_idx = f.create_dataset('good_idx', np.asarray(good_idx).shape, dtype="int")
    dset_good_idx[...] = np.asarray(good_idx)
    dset_etas = f.create_dataset('etas', etas.shape, dtype=dtype)
    dset_etas[...] = etas
    dset_pts = f.create_dataset('pts', pts.shape, dtype=dtype)
    dset_pts[...] = pts

    # rebuild pdf
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    ndata = np.sum(dataset,axis=-1)
    nsig = (1.-fbkg)*ndata
    nbkg = fbkg*ndata
    sigpdf = nsig[:,np.newaxis]*massWidth*kernelpdf(scaleBinned,sigmaBinned, datasetgen, masses,masses_gen)
    bkgpdf = nbkg[:,np.newaxis]*massWidth*exppdf(slope,masses)
    dset_sigpdf = f.create_dataset('sigpdf', sigpdf.shape, dtype=dtype)
    dset_sigpdf[...] = sigpdf
    dset_bkgpdf = f.create_dataset('bkgpdf', bkgpdf.shape, dtype=dtype)
    dset_bkgpdf[...] = bkgpdf

if debugPlots:
    plotsBkg(scaleBinned,scaleErrorBinned,sigmaBinned,fbkg,slope,dataset,datasetgen,masses,masses_gen,isJ,etas, good_idx, isData)

#have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
scaleplotBinned = ROOT.TH1D("scaleBinned", "scale", nBins, onp.linspace(0, nBins, nBins+1))
scaleplotBinned = array2hist(scaleBinned, scaleplotBinned, scaleErrorBinned)

sigmaplotBinned = ROOT.TH1D("sigmaBinned", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
sigmaplotBinned = array2hist(sigmaBinned, sigmaplotBinned, sigmaErrorBinned)

plots = [scaleplotBinned,sigmaplotBinned]

###### begin parameters fit
if fitCalibration:

    nModelParms = 6

    # fIn = ROOT.TFile.Open("calibrationDATAZ.root")
    # bHisto = fIn.Get("A")
    # A = hist2array(bHisto)

    A = np.zeros((nEtaBins),dtype=np.float64)
    e = np.zeros((nEtaBins),dtype=np.float64)
    M = np.zeros((nEtaBins),dtype=np.float64)
    a = 1e-6*np.ones((nEtaBins),dtype=np.float64)
    c = 10e-9*np.ones((nEtaBins),dtype=np.float64)
    b = np.zeros((nEtaBins),dtype=np.float64)
    d = 3.7*np.ones((nEtaBins),dtype=np.float64)

    xmodel = np.stack((A,e,M,a,c,b),axis=-1)

    chi2 = chi2LBins(xmodel, scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,binCenters1, binCenters2, good_idx)

    print(chi2)

    xmodel = pmin(chi2LBins, xmodel.flatten(), args=(scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx), doParallel=False)

    xmodel = xmodel.reshape((-1,nModelParms))

    fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
    hchi2 = jax.jit(jax.hessian(chi2LBins))

    chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,  binCenters1, binCenters2, good_idx)
    chi2hess = hchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,  binCenters1, binCenters2, good_idx)

    hmodel = chi2hess
    covmodel = np.linalg.inv(chi2hess)
    invhess = covmodel

    valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas,  binCenters1, binCenters2, good_idx)

    ndof = 2*nBins - nEtaBins*nModelParms
    edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

    print("nEtaBins", nEtaBins)
    print("nBins", nBins)
    print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

    errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

    A,e,M,a,b,c = modelParsFromParVector(xmodel)

    scaleSqModel = scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx)
    sigmaSqModel = sigmaSqFromModelPars(a, b, c, etas, binCenters1, binCenters2, good_idx)


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
    plt.savefig("corrmatrix{}.pdf".format("Data" if isData else "MC"))

    print("computing scales and errors:")

    ndata = np.sum(dataset,axis=-1)

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

if fitCalibration:
    if not isData:
        f = ROOT.TFile("calibrationZMC{}_aftersm.root".format(closure_flag), 'recreate')
    else:
        f = ROOT.TFile("calibrationZDATA{}_aftersm.root".format(closure_flag), 'recreate')
    f.cd()

    hA.Write()
    he.Write()
    hM.Write()
    ha.Write()
    hc.Write()
    hb.Write()
    hd.Write()

    correlationHist = ROOT.TH2D('correlation_matrix', 'correlation matrix', 48*nModelParms, 0., 1., 48*nModelParms, 0., 1.)
    covarianceHist  = ROOT.TH2D('covariance_matrix', 'covariance matrix', 48*nModelParms, 0., 1., 48*nModelParms, 0., 1.)
    correlationHist.GetZaxis().SetRangeUser(-1., 1.)

    array2hist(corr, correlationHist)
    array2hist(invhess, covarianceHist)

    correlationHist.Write()
    covarianceHist.Write()

    for plot in plots:
        plot.Write()

# scale and sigma plots in pdf

bins=np.linspace(1, scaleBinned.shape[0]+1, scaleBinned.shape[0]+1)
binsC = (bins[:-1] + bins[1:]) / 2
fig, ax1 = plt.subplots()
ax1.set_title("scale_{}".format("data" if isData else "mc"), fontsize=18)
hep.histplot(scaleBinned,bins,yerr=scaleErrorBinned,histtype = 'errorbar', ax=ax1, label = ["binned scale fit"])

if fitCalibration:
    ax1.plot(binsC, scaleModel, color="red", label = "fit to model", zorder=10)
    ax1.fill_between(binsC, scaleModel-scaleSigmaErrsModel[:,0], scaleModel+scaleSigmaErrsModel[:,0], color="red", alpha=0.4)

ax1.set_ylim([np.min(scaleBinned)-0.0005, np.max(scaleBinned)+0.0005])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig("scale_{}_{}{}.pdf".format("J" if isJ else "Z", "data" if isData else "MC",closure_flag))
plt.clf()

fig, ax1 = plt.subplots()
ax1.set_title("sigma_{}".format("data" if isData else "mc"), fontsize=18)
hep.histplot(sigmaBinned,bins,yerr=sigmaErrorBinned,histtype = 'errorbar', ax=ax1, label = ["binned sigma fit"])

if fitCalibration:
    ax1.plot(binsC, sigmaModel, color="red", label = "fit to model", zorder=10)
    ax1.fill_between(binsC, sigmaModel-scaleSigmaErrsModel[:,1], sigmaModel+scaleSigmaErrsModel[:,1], color="red", alpha=0.4)

ax1.set_ylim([np.min(sigmaBinned)-0.001, np.max(sigmaBinned)+0.001])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig("sigma_{}_{}.png".format("J" if isJ else "Z", "data" if isData else "MC"))
plt.clf()

# if debugPlots and fitCalibration:

#     files = glob.glob('/PlotsScale/*')
#     for f in files:
#         os.remove(f)

#     # separate plots per eta bins
#     full = onp.zeros((nEtaBins,nEtaBins,20,20),dtype='float64')
#     full[good_idx] = scaleBinned
#     scaleBinnedPatched = full

#     full = onp.zeros((nEtaBins,nEtaBins,20,20),dtype='float64')
#     full[good_idx] = scaleErrorBinned
#     scaleErrorBinnedPatched = full

#     full = onp.zeros((nEtaBins,nEtaBins,20,20),dtype='float64')
#     full[good_idx] = scaleModel
#     scaleModelPatched = full

#     full = onp.zeros((nEtaBins,nEtaBins,20,20),dtype='float64')
#     full[good_idx] = scaleSigmaErrsModel[:,0]
#     scaleErrsModelPatched = full

#     etasC = (etas[:-1] + etas[1:]) / 2
#     for ieta1 in range(len(etasC)):
#         for ieta2 in range(len(etasC)):
#             if np.all(scaleBinnedPatched[ieta1,ieta2,:,:]==0):
#                 continue
#             kbins=np.linspace(1, scaleBinnedPatched[ieta1,ieta2,:,:][scaleBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel().shape[0]+1, scaleBinnedPatched[ieta1,ieta2,:,:][scaleBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel().shape[0]+1)
#             kC = (kbins[:-1] + kbins[1:]) / 2
#             fig, ax1 = plt.subplots()
#             ax1.set_title("scale {} {}".format(round(etasC[ieta1],2),round(etasC[ieta2],2)), fontsize=18)
#             ax1.text(0.95, 0.1, 'A={}, e={}, M={}'.format(round(A[ieta1],5),round(e[ieta1],5),round(M[ieta1],5)),verticalalignment='bottom', horizontalalignment='right',transform=ax1.transAxes,color='black', fontsize=10)
#             hep.histplot(scaleBinnedPatched[ieta1,ieta2,:,:][scaleBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel(),kbins,yerr=scaleErrorBinnedPatched[ieta1,ieta2,:,:][scaleErrorBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel(),histtype = 'errorbar', ax=ax1, label = ["binned scale fit"])
#             # hep.histplot(scaleModelPatched[ieta1,ieta2,:,:].ravel(),kbins,yerr=scaleErrsModelPatched[ieta1,ieta2,:,:].ravel(),color = "r",histtype = 'errorbar', ax=ax1, label = ["fit to model"])
#             ax1.plot(kC, scaleModelPatched[ieta1,ieta2,:,:][scaleModelPatched[ieta1,ieta2,:,:]!=0.].ravel(), color="red", label = "fit to model")
#             ax1.fill_between(kC, (scaleModelPatched[ieta1,ieta2,:,:]-scaleErrsModelPatched[ieta1,ieta2,:,:])[scaleModelPatched[ieta1,ieta2,:,:]!=0.].ravel(), (scaleModelPatched[ieta1,ieta2,:,:]+scaleErrsModelPatched[ieta1,ieta2,:,:])[scaleModelPatched[ieta1,ieta2,:,:]!=0.].ravel(), color="red", alpha=0.4)
#             ax1.set_ylim([np.min(scaleBinnedPatched[ieta1,ieta2,:,:][scaleBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel())-0.0005, np.max(scaleBinnedPatched[ieta1,ieta2,:,:][scaleBinnedPatched[ieta1,ieta2,:,:]!=0.].ravel())+0.0005])
#             ax1.legend(loc='upper right', frameon=True)
#             ax1.set_xticklabels([])
#             plt.tight_layout()
#             plt.savefig("PlotsScale/scale_{}_{}.png".format(round(etasC[ieta1],2),round(etasC[ieta2],2)))
#             plt.clf()
#             # assert(0)

