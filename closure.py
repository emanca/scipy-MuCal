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
from jax import random

from obsminimization import pmin

import ROOT
ROOT.gROOT.SetBatch(True)
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
from scipy.stats import chisquare
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])
import itertools
import math
import array
from root_numpy import array2hist, hist2array, fill_hist

def sigmaSqFromSingleRes(x,good_idx):
    #compute sigma from single resolutions
    x = x.reshape((24,-1))

    res1 = x[good_idx[0],good_idx[2]] # eta1 and pt1
    res2 = x[good_idx[1],good_idx[3]] # eta2 and pt2
    
    sigmaSq = 0.25*(res1+res2)
    
    return sigmaSq

def sigmaSqFromModelParsSingleMu(a,b,c,d, binCenters, good_idx):
    
    #compute sigma from physics parameters
    a = a[good_idx[0]]
    b = b[good_idx[0]]
    c = c[good_idx[0]]
    d = d[good_idx[0]]

    pt2 = binCenters[...,2]
    L2 = binCenters[...,1]
    invpt2 = binCenters[...,3]
    sigmaSq = a*L2 + c*pt2*np.square(L2) + b*L2*np.reciprocal(1.+d*invpt2/L2)
    return sigmaSq

def sigmaSqFromModelParsSingleMuFromVec(x,binCenters, good_idx):

    a = x[...,0]
    b = x[...,1]
    c = x[...,2]
    d = x[...,3]

    return sigmaSqFromModelParsSingleMu(a,b,c,d, binCenters, good_idx)

def chi2LBins(x, sigmaSq, sigmaSqErr, good_idx):
    
    sigmaSqModel = sigmaSqFromSingleRes(x,good_idx)
    
    diff = sigmaSqModel-sigmaSq
    # print(sigmaSqModel, sigmaSq, diff)

    #batched column vectors
    diffcol = np.expand_dims(diff,-1)
    
    #batched row vectors
    diffcolT = np.expand_dims(diff,-2)

    #batched matrix multiplication
    hSigmaSq = np.diag(np.reciprocal(np.square(sigmaSqErr)))
    lbins = 0.5*np.matmul(diffcolT,np.matmul(hSigmaSq, diffcol))
    
    return np.sum(lbins)+0.5*np.sum(np.square((x - 0.0001)/0.5))

def chi2LBinsModel(x, sigmaSqJ, sigmaSqErrJ, binCentersJ, good_idxJ, sigmaSqZ, sigmaSqErrZ, binCentersZ, good_idxZ):
    
    x = x.reshape((-1,4))
    
    a = x[...,0]
    b = x[...,1]
    c = x[...,2]
    d = x[...,3]
    
    sigmaSqModelJ = sigmaSqFromModelParsSingleMu(a,b,c,d, binCentersJ, good_idxJ)
    sigmaSqModelZ = sigmaSqFromModelParsSingleMu(a,b,c,d, binCentersZ, good_idxZ)

    diffJ = sigmaSqModelJ-sigmaSqJ
    diffZ = sigmaSqModelZ-sigmaSqZ

    #batched column vectors
    diffcolJ = np.expand_dims(diffJ,-1)
    #batched row vectors
    diffcolTJ = np.expand_dims(diffJ,-2)

    #batched matrix multiplication
    hSigmaSqJ = np.diag(np.reciprocal(np.square(sigmaSqErrJ)))
    lbinsJ = 0.5*np.matmul(diffcolTJ,np.matmul(hSigmaSqJ, diffcolJ))

    #batched column vectors
    diffcolZ = np.expand_dims(diffZ,-1)
    #batched row vectors
    diffcolTZ = np.expand_dims(diffZ,-2)

    #batched matrix multiplication
    hSigmaSqZ = np.diag(np.reciprocal(np.square(sigmaSqErrZ)))
    lbinsZ = 0.5*np.matmul(diffcolTZ,np.matmul(hSigmaSqZ, diffcolZ))
    
    return np.sum(lbinsJ)+np.sum(lbinsZ)

# fileJPsiMC = h5py.File('fitsJMC_sigma.hdf5', mode='r')
# fileZMC = h5py.File('fitsZMC_sigma.hdf5', mode='r')

fileJPsiMC = h5py.File('fitsJDATA.hdf5', mode='r')
fileZMC = h5py.File('fitsZDATA.hdf5', mode='r')

etas = fileJPsiMC['etas'][:]
etasC = (etas[:-1] + etas[1:]) / 2

ptsJ =np.array([ 3.,3.85838168,5.2,6.25981158,7.95232633, 25.        ])
ptsJC = (ptsJ[:-1] + ptsJ[1:]) / 2
nPtBinsJ = ptsJ.shape[0]-1

sigmaSqJ = fileJPsiMC['sigmaSq'][:]
sigmaSqErrJ = fileJPsiMC['sigmaSqErr'][:]
good_idxJ = tuple(fileJPsiMC['good_idx'][:])

sigmaSqZ = fileZMC['sigmaSq'][:]
sigmaSqErrZ = fileZMC['sigmaSqErr'][:]
good_idxZ = fileZMC['good_idx'][:]

ptsZ = np.array([30.,35., 39., 43, 48., 70.])
ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
nPtBinsZ = ptsZ.shape[0]-1

nEtaBins = etas.shape[0]-1
print(nEtaBins)
nBinsJ = sigmaSqJ.shape[0]
nBinsZ = sigmaSqZ.shape[0]

sigmaSqJPatched = onp.zeros((nEtaBins,nEtaBins,nPtBinsJ,nPtBinsJ),dtype='float64')
# print(sigmaSqJPatched.shape,sigmaSqJ.shape, len(good_idxJ),good_idxJ[0].shape,sigmaSqJPatched[good_idxJ].shape)
sigmaSqJPatched[good_idxJ] = sigmaSqJ

good_idxJ = np.nonzero(sigmaSqJPatched>0.0015**2)
sigmaSqJ = sigmaSqJ[sigmaSqJ>0.0015**2]
sigmaSqErrJ = sigmaSqJ[sigmaSqJ>0.0015**2]

# # fitting J
x=0.0001*np.ones((nEtaBins,nPtBinsJ))
chi2 = chi2LBins(x, sigmaSqJ, sigmaSqErrJ, good_idxJ)
print(chi2)

xmodelJ = pmin(chi2LBins, x.flatten(), args=(sigmaSqJ, sigmaSqErrJ, good_idxJ), doParallel=False)
xmodelJ = xmodelJ.reshape((nEtaBins,nPtBinsJ))

fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
hchi2 = jax.jit(jax.hessian(chi2LBins))

chi2,chi2grad = fgchi2(xmodelJ.flatten(), sigmaSqJ, sigmaSqErrJ, good_idxJ)
chi2hess = hchi2(xmodelJ.flatten(), sigmaSqJ, sigmaSqErrJ, good_idxJ)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

valmodel,gradmodel = fgchi2(xmodelJ.flatten(), sigmaSqJ, sigmaSqErrJ, good_idxJ)

ndof = 2*(nBinsJ) - nEtaBins
edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

print("nEtaBins", nEtaBins)
print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

errsmodelJ = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsJ))
print(xmodelJ, '+/-', errsmodelJ)

# fitting Z
x=0.0001*np.ones((nEtaBins,nPtBinsZ))
chi2 = chi2LBins(x, sigmaSqZ, sigmaSqErrZ, good_idxZ)


xmodelZ = pmin(chi2LBins, x.flatten(), args=(sigmaSqZ, sigmaSqErrZ, good_idxZ), doParallel=False)
xmodelZ = xmodelZ.reshape((nEtaBins,nPtBinsZ))

fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
hchi2 = jax.jit(jax.hessian(chi2LBins))

chi2,chi2grad = fgchi2(xmodelZ.flatten(), sigmaSqZ, sigmaSqErrZ, good_idxZ)
chi2hess = hchi2(xmodelZ.flatten(), sigmaSqZ, sigmaSqErrZ, good_idxZ)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

valmodel,gradmodel = fgchi2(xmodelZ.flatten(), sigmaSqZ, sigmaSqErrZ, good_idxZ)

ndof = 2*(nBinsZ) - nEtaBins
edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

print("nEtaBins", nEtaBins)
print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

errsmodelZ = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsZ))
print(xmodelZ, '+/-', errsmodelZ)

sigmaJ = np.sqrt(xmodelJ)
sigmaJError = 0.5*errsmodelJ/sigmaJ

sigmaZ = np.sqrt(xmodelZ)
sigmaZError = 0.5*errsmodelZ/sigmaZ

fileZMC = h5py.File('JPsiInputData/ZMC_mukin.hdf5', mode='r')
harrayZpl = fileZMC['Jpsi_distr_mcplus'][:]
harrayZplmeans = fileZMC['Jpsi_distr_mcplus_means'][:]
# harrayZplgen = fileZMC['Jpsi_distr_mcplusgen'][:]
# harrayZplmeansgen = fileZMC['Jpsi_distr_mcplus_meansgen'][:]

fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')
harraypl = fileJPsiMC['Jpsi_distr_mcplus'][:]
harrayJplmeans = fileJPsiMC['Jpsi_distr_mcplus_means'][:]

good_idxJ = np.nonzero(np.sum(harraypl,axis=(-1)).T>0.)
good_idxZ = np.nonzero(np.sum(harrayZpl,axis=(-1)).T>0.)

bincenter_idx = [4,0,6,8,2]
binCentersJfull = np.swapaxes((harrayJplmeans/np.expand_dims(np.sum(harraypl,axis=-1),axis=-1)),0,1)[...,bincenter_idx]
binCentersZfull = np.swapaxes((harrayZplmeans/np.expand_dims(np.sum(harrayZpl,axis=-1),axis=-1)),0,1)[...,bincenter_idx]
# binCentersZfullgen = np.swapaxes((harrayZplmeansgen/np.expand_dims(np.sum(harrayZplgen,axis=-1),axis=-1)),0,1)[...,bincenter_idx]


# ### parameters fit

# binCentersJ = binCentersJfull[good_idxJ]
# binCentersZ = binCentersZfull[good_idxZ]
# nModelParms = 4

# a = 1e-6*np.ones((nEtaBins),dtype=np.float64)
# c = 10e-9*np.ones((nEtaBins),dtype=np.float64)
# b = np.zeros((nEtaBins),dtype=np.float64)
# d = 100*np.ones((nEtaBins),dtype=np.float64)

# xmodel = np.stack((a,b,c,d),axis=-1)
# chi2 = chi2LBinsModel(xmodel.flatten(), xmodelJ[good_idxJ].flatten(), errsmodelJ[good_idxJ].flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# print(chi2)
# xmodel = pmin(chi2LBinsModel, xmodel.flatten(), args=(xmodelJ[good_idxJ].flatten(), errsmodelJ[good_idxJ].flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ), doParallel=False)
# xmodel = xmodel.reshape((-1,4))

# fgchi2 = jax.jit(jax.value_and_grad(chi2LBinsModel))
# hchi2 = jax.jit(jax.hessian(chi2LBinsModel))

# chi2,chi2grad = fgchi2(xmodel.flatten(), xmodelJ[good_idxJ].flatten(), errsmodelJ[good_idxJ].flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)
# chi2hess = hchi2(xmodel.flatten(), xmodelJ[good_idxJ].flatten(), errsmodelJ[good_idxJ].flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# hmodel = chi2hess
# covmodel = np.linalg.inv(chi2hess)
# invhess = covmodel

# valmodel,gradmodel = fgchi2(xmodel.flatten(), xmodelJ[good_idxJ].flatten(), errsmodelJ[good_idxJ].flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# ndof = 2*(nBinsJ+nBinsZ) - nEtaBins*nModelParms
# edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

# print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

# errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

# a = xmodel[...,0]
# b = xmodel[...,1]
# c = xmodel[...,2]
# d = xmodel[...,3]

# sigmaSqModelJ = sigmaSqFromModelParsSingleMu(a,b,c,d, binCentersJ, good_idxJ)
# sigmaModelJFull=onp.zeros((nEtaBins,nPtBinsJ))
# sigmaModelJFull[good_idxJ]= np.sqrt(sigmaSqModelJ)
# sigmaModelJ = sigmaModelJFull.reshape(nEtaBins,nPtBinsJ)

# sigmaSqModelZ = sigmaSqFromModelParsSingleMu(a,b,c,d, binCentersZ, good_idxZ)
# sigmaModelZ = np.sqrt(sigmaSqModelZ).reshape(nEtaBins,nPtBinsZ)

# sigmajac = jax.jit(jax.jacfwd(sigmaSqFromModelParsSingleMuFromVec))(xmodel, binCentersJ, good_idxJ)
# sigmajac = np.reshape(sigmajac, (-1,covmodel.shape[0]))
# covSigmaModel = np.matmul(sigmajac,np.matmul(covmodel,sigmajac.T))
# SigmaErrsModelJ = np.sqrt(np.diag(covSigmaModel))
# sigmaModelJFull=onp.zeros((nEtaBins,nPtBinsJ))
# sigmaModelJFull[good_idxJ]= np.sqrt(sigmaSqModelJ)
# SigmaErrsModelJ = sigmaModelJFull.reshape(nEtaBins,nPtBinsJ)

# sigmajac = jax.jit(jax.jacfwd(sigmaSqFromModelParsSingleMuFromVec))(xmodel, binCentersZ, good_idxZ)
# sigmajac = np.reshape(sigmajac, (-1,covmodel.shape[0]))
# covSigmaModel = np.matmul(sigmajac,np.matmul(covmodel,sigmajac.T))
# SigmaErrsModelZ = np.sqrt(np.diag(covSigmaModel))
# SigmaErrsModelZ = SigmaErrsModelZ.reshape(nEtaBins,nPtBinsZ)

# print(xmodel, "+/-", errsmodel)
# print(edm, "edm")

# diag = np.diag(np.sqrt(np.diag(invhess)))
# diag = np.linalg.inv(diag)
# corr = np.dot(diag,invhess).dot(diag)

# # plt.clf()
# # plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
# # plt.colorbar()
# # plt.savefig("corrmatrix{}.pdf".format("Data" if isData else "MC"))

# print("computing scales and errors:")

# aerr = errsmodel[:,0]
# berr = errsmodel[:,1]
# cerr = errsmodel[:,2]
# derr = errsmodel[:,3]

f = ROOT.TFile("sigmaDATA.root", 'recreate')

for i in range(nEtaBins):
    # if not round(etasC[i],2)==1.1: continue
    fig, (ax1) = plt.subplots()
    # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
    hep.cms.text('work in progress', ax=ax1)
    # ax1.set_title("scale", fontsize=18)
    # ax1.text(0.95, 0.95, 'a: {:.5f}+/-{:.6f}\n b: {:.5f}+/-{:.6f}\n c: {:.10f}+/-{:.11f}\n d: {:.1f}+/-{:.1f}\n'\
    #                     .format(a[i],aerr[i],b[i],berr[i],c[i],cerr[i],d[i],derr[i]),
    #     verticalalignment='top', horizontalalignment='right',
    #     transform=ax1.transAxes,
    #     color='black', fontsize=10)
    ax1.text(0.95, 0.95,'eta {}'.format(etasC[i]),verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    binstot = onp.concatenate((binCentersJfull[i,...,2][~onp.isnan(binCentersJfull[i,...,2])],binCentersZfull[i,...,2]))
    # xnew = np.linspace(binstot.min(), binstot.max(), 100) 
    # modeltot = np.concatenate((sigmaModelJ[i,...][sigmaModelJ[i,...]>0.],sigmaModelZ[i,...]))
    # errmodeltot = np.concatenate((SigmaErrsModelJ[i,...][SigmaErrsModelJ[i,...]>0.],SigmaErrsModelZ[i,...]))
    # spl = make_interp_spline(binstot, modeltot, k=3)  # type: BSpline
    # power_smooth = spl(xnew)
    # splUp = make_interp_spline(binstot, modeltot+errmodeltot, k=3)  # type: BSpline
    # power_smoothUp = spl(xnew)
    # splDown = make_interp_spline(binstot, modeltot-errmodeltot, k=3)  # type: BSpline
    # power_smoothDown = spl(xnew)
    ax1.errorbar(binCentersJfull[i,...,2],sigmaJ[i,...], yerr=sigmaJError[i,...], marker="v", label = '$J/\psi$ mc',fmt='v')
    ax1.errorbar(binCentersZfull[i,...,2],sigmaZ[i,...], yerr=sigmaZError[i,...], marker="v", label = '$Z$ mc',fmt='v')
    # ax1.plot(binstot,modeltot, color="red")
    # ax1.plot(xnew,power_smooth, color="red")
    # ax1.fill_between(xnew,power_smoothDown,power_smoothUp, color="red", alpha=0.5)
    ax1.set_ylabel('$\sigma_{p_T}/{p_T}$')
    # ax1.set_ylabel('$\sigma_{p_T}$')
    ax1.set_xlabel('$p_T^2$')
    ax1.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('PlotsSigma/sigma_eta{}.png'.format(round(etasC[i],2)))

    f.cd()
    proj = ROOT.TGraphErrors(binstot.shape[0],binstot,onp.concatenate((sigmaJ[i,2:],sigmaZ[i,...])),onp.zeros_like(binstot),onp.concatenate((sigmaJError[i,2:],sigmaZError[i,...])))
    proj.SetName("sigma_bineta{}".format(i))
    proj.Write()

etaarr = onp.array(etas.tolist())
ha = ROOT.TH1D("a", "a", nEtaBins, etaarr)
hc = ROOT.TH1D("c", "c", nEtaBins, etaarr)
hb = ROOT.TH1D("b", "b", nEtaBins, etaarr)
hd = ROOT.TH1D("d", "d", nEtaBins, etaarr)

ha = array2hist(a, ha, aerr)
hc = array2hist(c, hc, cerr)
hb = array2hist(b, hb, berr)
hd = array2hist(d, hd, derr)

ha.GetYaxis().SetTitle('material correction (resolution) a^2')
hc.GetYaxis().SetTitle('hit position (resolution) c^2')

ha.GetXaxis().SetTitle('#eta')
hc.GetXaxis().SetTitle('#eta')


ha.Write()
hc.Write()
hb.Write()
hd.Write()

correlationHist = ROOT.TH2D('correlation_matrix', 'correlation matrix', 10*nModelParms, 0., 1., 10*nModelParms, 0., 1.)
covarianceHist  = ROOT.TH2D('covariance_matrix', 'covariance matrix', 10*nModelParms, 0., 1., 10*nModelParms, 0., 1.)
correlationHist.GetZaxis().SetRangeUser(-1., 1.)

array2hist(corr, correlationHist)
array2hist(invhess, covarianceHist)

correlationHist.Write()
covarianceHist.Write()

# for plot in plots:
#     plot.Write()