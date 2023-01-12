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
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])
import itertools
import math
import array

def scaleSqFromSingleRes(x,good_idx):
    #compute scale from single resolutions
    x = x.reshape((24,5))

    # etas = np.linspace(-2.4,2.4,25)
    # etasC = (etas[:-1] + etas[1:]) / 2
    # etasRed =  np.linspace(-2.4,2.4,2)
    # bineta1 = np.digitize(etasC[good_idx[0]], etasRed)-1
    # bineta2 = np.digitize(etasC[good_idx[1]], etasRed)-1

    # ptsZ = np.array([30.,35., 39., 43, 48., 70.])
    # ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
    # ptsZRed = np.array([30., 70.])
    # bineta1 = np.digitize(ptsZC[good_idx[2]], ptsZRed)-1
    # bineta2 = np.digitize(ptsZC[good_idx[3]], ptsZRed)-1

    # res1 = x[bineta1,good_idx[2]] # eta1 and pt1
    # res2 = x[bineta2,good_idx[3]] # eta2 and pt2

    # res1 = x[good_idx[0],bineta1] # eta1 and pt1
    # res2 = x[good_idx[1],bineta2] # eta2 and pt2

    res1 = x[good_idx[0],good_idx[2]] # eta1 and pt1
    res2 = x[good_idx[1],good_idx[3]] # eta2 and pt2

    scaleSq = res1*res2
    # scaleSq =np.log(res1)+np.log(res2)
    
    return scaleSq

def scaleSqFromModelParsSingleMu(A,e,M, binCenters, good_idx):
    
    #compute scale from physics parameters

    A = A[good_idx[0]]
    e = e[good_idx[0]]
    M = M[good_idx[0]]

    coeffe = binCenters[...,0]
    coeffM = binCenters[...,1]
    
    term = 1 + A + e*coeffe +M*coeffM
    
    scaleSq = np.square(1.-term1)
        
    return scaleSq

def chi2LBins(x, scaleSq, scaleSqErr, good_idx):
    
    scaleSqModel = scaleSqFromSingleRes(x,good_idx)
    diff = scaleSqModel-scaleSq
    # print(scaleSqModel, scaleSq, diff)

    #batched column vectors
    diffcol = np.expand_dims(diff,-1)
    
    #batched row vectors
    diffcolT = np.expand_dims(diff,-2)

    #batched matrix multiplication
    hScaleSq = np.diag(np.reciprocal(np.square(scaleSqErr)))
    lbins = 0.5*np.matmul(diffcolT,np.matmul(hScaleSq, diffcol))
    
    return np.sum(lbins)+0.5*np.sum(np.square((x - 1.)/0.05))

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

fileJPsiMC = h5py.File('fitsJMC.hdf5', mode='r')
fileZMC = h5py.File('fitsZMC_corr_50sm.hdf5', mode='r')
# fileZMC = h5py.File('fitsJDATA_corr.hdf5', mode='r')

scaleSqJ = fileJPsiMC['scaleSq'][:]
scaleSqErrJ = fileJPsiMC['scaleSqErr'][:]
good_idxJ = fileJPsiMC['good_idx'][:]

# etas = fileJPsiMC['etas'][:]
etas = np.linspace(-2.4,2.4,25)
etasC = (etas[:-1] + etas[1:]) / 2

ptsJ =np.array([ 2.,3.9,5.2,6.3,8.7,25.])
ptsJC = (ptsJ[:-1] + ptsJ[1:]) / 2
nPtBinsJ = ptsJ.shape[0]-1

scaleSqZ = fileZMC['scaleSq'][:]
scaleSqErrZ = fileZMC['scaleSqErr'][:]
good_idxZ = fileZMC['good_idx'][:]

# bound1 =  np.digitize(-1.5, etas)-1
# bound2 =  np.digitize(1.5, etas)-1

# print(bound1,bound2)

# full = -99.*onp.ones((24,24,5,5))
# full[tuple(good_idxZ)]=scaleSqZ

# # scaleSqZ = full[bound1+1:bound2,bound1+1:bound2,...][full[bound1+1:bound2,bound1+1:bound2,...]>0.].ravel()
# scaleSqZ = full[:bound1+1,:bound1+1,...][full[:bound1+1,:bound1+1,...]>0.].ravel()
# # scaleSqZ = full[bound2:,bound2:,...][full[bound2:,bound2:,...]>0.].ravel()

# full = -99.*onp.ones((24,24,5,5))
# full[tuple(good_idxZ)]=scaleSqErrZ
# # scaleSqErrZ = full[bound1+1:bound2,bound1+1:bound2,...][full[bound1+1:bound2,bound1+1:bound2,...]>0.].ravel()
# scaleSqErrZ = full[:bound1+1,:bound1+1,...][full[:bound1+1,:bound1+1,...]>0.].ravel()
# # scaleSqErrZ = full[bound2:,bound2:,...][full[bound2:,bound2:,...]>0.].ravel()

# # good_idxZ = np.nonzero(full[bound1+1:bound2,bound1+1:bound2,...]>0.)
# good_idxZ = np.nonzero(full[:bound1+1,:bound1+1,...]>0.)
# # good_idxZ = np.nonzero(full[bound2:,bound2:,...]>0.)
# print(len(good_idxZ[0]))

# etas = np.linspace(-2.4,2.4,2)
# etasC = (etas[:-1] + etas[1:]) / 2
# etasRed =  np.linspace(-2.4,2.4,3)
# bineta1 = np.digitize(etasC[good_idxZ[0]], etasRed)-1
# bineta2 = np.digitize(etasC[good_idxZ[1]], etasRed)-1
# print(bineta1)
# print(bineta2)

# good_idxZ_sel = np.nonzero(full>0.)
# scaleSqZ = full[tuple(good_idxZ_sel)]

# full = -99.*onp.ones((2,2,2,2))
# full[tuple(good_idxZ)]=scaleSqErrZ

# scaleSqErrZ = full[tuple(good_idxZ_sel)]
# good_idxZ = good_idxZ_sel

ptsZ = np.array([30.,35., 39., 43, 48., 70.])
# ptsZ = np.array([30.,70.])
ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
nPtBinsZ = ptsZ.shape[0]-1

nEtaBins = etas.shape[0]-1
nBinsJ = scaleSqJ.shape[0]
nBinsZ = scaleSqZ.shape[0]

# fitting J
x=np.ones((nEtaBins,nPtBinsJ))
chi2 = chi2LBins(x, scaleSqJ, scaleSqErrJ, good_idxJ)
print(chi2)

xmodelJ = pmin(chi2LBins, x.flatten(), args=(scaleSqJ, scaleSqErrJ, good_idxJ), doParallel=False)
xmodelJ = xmodelJ.reshape((nEtaBins,nPtBinsJ))

fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
hchi2 = jax.jit(jax.hessian(chi2LBins))

chi2,chi2grad = fgchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)
chi2hess = hchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

valmodel,gradmodel = fgchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)

ndof = 2*(nBinsJ) - nEtaBins
edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

print("nEtaBins", nEtaBins)
print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

errsmodelJ = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsJ))
print(xmodelJ, '+/-', errsmodelJ)

# # fitting Z
# x=np.ones((nEtaBins,nPtBinsZ))
# chi2 = chi2LBins(x, scaleSqZ, scaleSqErrZ, good_idxZ)

# print(chi2)
# # assert(0)

# xmodelZ = pmin(chi2LBins, x.flatten(), args=(scaleSqZ, scaleSqErrZ, good_idxZ), doParallel=False)
# xmodelZ = xmodelZ.reshape((nEtaBins,nPtBinsZ))

# fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
# hchi2 = jax.jit(jax.hessian(chi2LBins))

# chi2,chi2grad = fgchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)
# chi2hess = hchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)

# hmodel = chi2hess
# covmodel = np.linalg.inv(chi2hess)
# invhess = covmodel

# valmodel,gradmodel = fgchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)

# ndof = nBinsZ - nEtaBins*nPtBinsZ*2
# edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

# print("nEtaBins", nEtaBins)
# print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

# errsmodelZ = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsZ))
# print(xmodelZ, '+/-', errsmodelZ)

scaleJ = xmodelJ
scaleJError = errsmodelJ

# scaleZ = xmodelZ
# scaleZError = errsmodelZ

# diag = np.diag(np.sqrt(np.diag(invhess)))
# diag = np.linalg.inv(diag)
# corr = np.dot(diag,invhess).dot(diag)

# import matplotlib
# # matplotlib.use('agg')
# import matplotlib.pyplot as plt

# plt.clf()
# plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
# plt.colorbar()
# plt.show()
# # plt.savefig("corrmatrix.pdf")


# fileZMC = h5py.File('JPsiInputData/ZMC_mukin.hdf5', mode='r')
# harrayZpl = fileZMC['Jpsi_distr_mcplus'][:]
# harrayZplmeans = fileZMC['Jpsi_distr_mcplus_means'][:]

# fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')
# harraypl = fileJPsiMC['Jpsi_distr_mcplus'][:]
# harrayJplmeans = fileJPsiMC['Jpsi_distr_mcplus_means'][:]

# good_idxJ = np.nonzero(np.sum(harraypl,axis=(-1)).T>0.)
# good_idxZ = np.nonzero(np.sum(harrayZpl,axis=(-1)).T>0.)

# bincenter_idx = [2,4]
# binCentersJfull = np.swapaxes((harrayJplmeans/np.expand_dims(np.sum(harraypl,axis=-1),axis=-1)),0,1)[...,bincenter_idx]
# binCentersZfull = np.swapaxes((harrayZplmeans/np.expand_dims(np.sum(harrayZpl,axis=-1),axis=-1)),0,1)[...,bincenter_idx]

# ### parameters fit

# binCentersJ = binCentersJfull[good_idxJ]
# binCentersZ = binCentersZfull[good_idxZ]
# nModelParms = 4

# A = np.zeros((nEtaBins),dtype=np.float64)
# e = np.zeros((nEtaBins),dtype=np.float64)
# M = np.zeros((nEtaBins),dtype=np.float64)

# xmodel = np.stack((A,e,M),axis=-1)
# chi2 = chi2LBinsModel(xmodel.flatten(), xmodelJ.flatten(), errsmodelJ.flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# print(chi2)

# xmodel = pmin(chi2LBinsModel, xmodel.flatten(), args=(xmodelJ.flatten(), errsmodelJ.flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ), doParallel=False)
# xmodel = xmodel.reshape((-1,3))

# fgchi2 = jax.jit(jax.value_and_grad(chi2LBinsModel))
# hchi2 = jax.jit(jax.hessian(chi2LBinsModel))

# chi2,chi2grad = fgchi2(xmodel.flatten(), xmodelJ.flatten(), errsmodelJ.flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)
# chi2hess = hchi2(xmodel.flatten(), xmodelJ.flatten(), errsmodelJ.flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# hmodel = chi2hess
# covmodel = np.linalg.inv(chi2hess)
# invhess = covmodel

# valmodel,gradmodel = fgchi2(xmodel.flatten(), xmodelJ.flatten(), errsmodelJ.flatten(), binCentersJ, good_idxJ, xmodelZ.flatten(), errsmodelZ.flatten(), binCentersZ, good_idxZ)

# ndof = 2*(nBinsJ+nBinsZ) - nEtaBins*nModelParms
# edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

# print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

# errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

# A = xmodel[...,0]
# e = xmodel[...,1]
# M = xmodel[...,2]

# scaleSqModelJ = scaleSqFromModelParsSingleMu(a,b,c,d, binCentersJ, good_idxJ)
# scaleModelJ = np.sqrt(scaleSqModelJ).reshape(nEtaBins,nPtBinsJ)

# scaleSqModelZ = scaleSqFromModelParsSingleMu(a,b,c,d, binCentersZ, good_idxZ)
# scaleModelZ = np.sqrt(scaleSqModelZ).reshape(nEtaBins,nPtBinsZ)

# # scalejac = jax.jit(jax.jacfwd(scaleSqFromModelParsSingleMu))(a,b,c,d, binCentersJ, good_idxJ)
# # print(scaleSqModelJ.shape,scalejac.shape,covmodel.shape)
# # # scalejac = np.reshape(scalejac, (-1,covmodel.shape[0]))
# # print(scalejac.shape)
# # covscaleModel = np.matmul(scalejac,np.matmul(covmodel,scalejac.T))
# # scaleErrsModelJ = np.sqrt(np.diag(covscaleModel))
# # scaleErrsModelJ = scaleErrsModelJ.reshape(nEtaBins,nPtBinsJ)

# # scalejac = jax.jit(jax.jacfwd(scaleSqFromModelParsSingleMu))(a,b,c,d, binCentersZ, good_idxZ)
# # scalejac = np.reshape(scalejac, (-1,covmodel.shape[0]))
# # covscaleModel = np.matmul(scalejac,np.matmul(covmodel,scalejac.T))
# # scaleErrsModelZ = np.sqrt(np.diag(covscaleModel))
# # scaleErrsModelZ = scaleErrsModelJ.reshape(nEtaBins,nPtBinsZ)

# print(xmodel, "+/-", errsmodel)
# print(edm, "edm")

# diag = np.diag(np.sqrt(np.diag(invhess)))
# diag = np.linalg.inv(diag)
# corr = np.dot(diag,invhess).dot(diag)

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

# # plt.clf()
# # plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
# # plt.colorbar()
# # plt.savefig("corrmatrix{}.pdf".format("Data" if isData else "MC"))

# print("computing scales and errors:")

# aerr = errsmodel[:,0]
# berr = errsmodel[:,1]
# cerr = errsmodel[:,2]
# derr = errsmodel[:,3]

for i in range(nEtaBins):
    fig, (ax1) = plt.subplots()
    # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
    hep.cms.text('work in progress', ax=ax1)
    ax1.text(0.95, 0.95,'$\eta$ {}'.format(etasC[i]),verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=24)
    # ax1.set_title("scale", fontsize=18)
    # ax1.text(0.95, 0.95, 'a: {:.5f}+/-{:.6f}\n b: {:.5f}+/-{:.6f}\n c: {:24f}+/-{:.11f}\n d: {:.1f}+/-{:.1f}\n'.format(a[i],aerr[i],b[i],berr[i],c[i],cerr[i],d[i],derr[i]),verticalalignment='top', horizontalalignment='right',transform=ax1.transAxes,color='black', fontsize24)
    ax1.errorbar(ptsJC[1:],scaleJ[i,...][1:], yerr=scaleJError[i,...][1:], marker="v", label = '$J/\psi$ mc',fmt='v')
    # ax1.errorbar(binCentersZfull[i,...,1],scaleZ[i,...],yerr=scaleZError[i,...], marker="v", label = '$Z$ mc',fmt='v')
    # ax1.plot(binCentersJfull[i,...,1],scaleModelJ[i,...], color="red")
    # ax1.fill_between(binCentersJfull[i,...,1],scaleModelJ[i,...]-scaleErrsModelJ[i,...],scaleModelJ[i,...]+scaleErrsModelJ[i,...], color="red", alpha=0.5)
    # ax1.plot(binCentersZfull[i,...,1],scaleModelZ[i,...], color="red")
    # ax1.fill_between(binCentersZfull[i,...,1],scaleModelZ[i,...]-scaleErrsModelZ[i,...],scaleModelZ[i,...]+scaleErrsModelZ[i,...], color="red", alpha=0.5)
    ax1.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('PlotsScale/scale_eta{}.png'.format(round(etasC[i],2)))

# integrate over one of the 2 variables
# integrate over eta
# scalePt = []
# scalePtErrs = []
# scalePtZ = []
# scalePtErrsZ = []
# for i in range(nPtBinsJ):
#     scalePt.append(onp.average(scaleJ[:,i],axis=0, weights=onp.reciprocal(onp.square(scaleJError[:,i]))))
#     scalePtErrs.append(onp.sqrt(1./onp.sum(onp.reciprocal(onp.square(scaleJError[:,i])))))
# for i in range(nPtBinsZ):
#     scalePtZ.append(onp.average(scaleZ[:,i],axis=0, weights=onp.reciprocal(onp.square(scaleZError[:,i]))))
#     scalePtErrsZ.append(onp.sqrt(1./onp.sum(onp.reciprocal(onp.square(scaleZError[:,i])))))
# print(scalePt,scalePtZ)

# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# ax1.errorbar(ptsJC,np.array(scalePt), yerr=np.array(scalePtErrs), marker="v", label = '$J/\psi$ mc',fmt='v')
# ax1.errorbar(ptsZC,np.array(scalePtZ), yerr=np.array(scalePtErrsZ), marker="v", label = '$Z$ mc',fmt='v')
# ax1.set_ylim([1-0.0005, 1.0005])
# ax1.set_xlabel('$p_T$')
# ax1.legend(loc='upper left', frameon=False)
# plt.tight_layout()
# plt.savefig('scale_pt_closure.png')

# # integrate over pt
# scaleEta = []
# scaleEtaErrs = []
# scaleEtaZ = []
# scaleEtaErrsZ = []
# for i in range(nEtaBins):
#     scaleEta.append(onp.average(scaleJ[i,:],axis=0, weights=onp.reciprocal(onp.square(scaleJError[i,:]))))
#     scaleEtaErrs.append(onp.sqrt(1./onp.sum(onp.reciprocal(onp.square(scaleJError[i,:])))))
#     scaleEtaZ.append(onp.average(scaleZ[i,:],axis=0, weights=onp.reciprocal(onp.square(scaleZError[i,:]))))
#     scaleEtaErrsZ.append(onp.sqrt(1./onp.sum(onp.reciprocal(onp.square(scaleZError[i,:])))))
# print(scaleEta)

# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# ax1.errorbar(etasC,np.array(scaleEta), yerr=np.array(scaleEtaErrs), marker="v", label = '$J/\psi$ mc',fmt='v')
# ax1.errorbar(etasC,np.array(scaleEtaZ), yerr=np.array(scaleEtaErrsZ), marker="v", label = '$Z$ mc',fmt='v')
# ax1.set_ylim([1-0.0005, 1.0005])
# ax1.set_xlabel('$\eta$')
# ax1.legend(loc='upper left', frameon=False)
# plt.tight_layout()
# plt.savefig('scale_eta_closure.png')

# save output
fileOut = "scale2D_Jmc_50sm.hdf5"

with h5py.File(fileOut, mode="w") as f:
    dtype = 'float64'
    dset_scale = f.create_dataset('scale', scaleZ.shape, dtype=dtype)
    dset_scale[...] = scaleZ
    dset_scale = f.create_dataset('scaleErr', scaleZError.shape, dtype=dtype)
    dset_scale[...] = scaleZError


# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# # hep.hist2dplot(round(np.array(scaleZ),4),etas,ptsZ, labels =True, cmap = 'jet')
# c = plt.pcolor(np.array(scaleZ), edgecolors='k', cmap='jet')

# def show_values(pc, fmt="%.5f", **kw):
#     pc.update_scalarmappable()
#     ax = pc.axes
#     for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
#         x, y = p.vertices[:-2, :].mean(0)
#         if np.all(color[:3] > 0.5):
#             color = (0.0, 0.0, 0.0)
#         else:
#             color = (1.0, 1.0, 1.0)
#         ax.text(x, y, fmt % value, ha="center", va="center",  fontsize=12, **kw)

# ax1.set_yticks(np.arange(len(etasC))+0.5)
# ax1.set_yticklabels(np.around(etasC,1))

# ax1.set_xticks(np.arange(len(ptsZC))+0.5)
# ax1.set_xticklabels(ptsZC)

# show_values(c)

# plt.colorbar(c)
# # ax1.set_ylim([1-0.0005, 1.0005])
# # ax1.set_xlabel('$p_T$')
# # ax1.legend(loc='upper left', frameon=False)
# plt.tight_layout()
# plt.savefig('scale_2dclosure.png')