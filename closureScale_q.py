import os
import glob
import multiprocessing
import h5py
import pickle
import lz4.frame

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

def scaleSqFromSingleRes(x,good_idx):
    #compute total scale from single scales
    x = x.reshape((1,8,2))

    etas = np.linspace(-2.4,2.4,9)
    etasC = (etas[:-1] + etas[1:]) / 2
    etasRed = np.array([-2.4,2.4])
    bineta1 = np.digitize(etasC[good_idx[0]], etasRed)-1
    bineta2 = np.digitize(etasC[good_idx[1]], etasRed)-1

    ptsZ = np.array([-3.14159265, -2.35619449, -1.57079633, -0.78539816,  0., 0.78539816,  1.57079633,  2.35619449,  3.14159265])
    ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
    ptsZRed = ptsZ
    binpts1 = np.digitize(ptsZC[good_idx[2]], ptsZRed)-1
    binpts2 = np.digitize(ptsZC[good_idx[3]], ptsZRed)-1

    res1 = np.sqrt(x[bineta1,binpts1,0]) # eta1 and pt1
    res2 = np.sqrt(x[bineta2,binpts2,1]) # eta2 and pt2
    
    scaleSq = np.square(res1*res2)
    
    return scaleSq

def scaleSqFromModelParsSingleMu(A,M, binCenters1, binCenters2, good_idx):
    
    coeffe1 = binCenters1[...,1]
    coeffM1 = binCenters1[...,0]

    coeffe2 = binCenters2[...,1]
    coeffM2 = binCenters2[...,0]

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0],good_idx[2]]
    # e1 = e[good_idx[0]]
    M1 = M[good_idx[0],good_idx[2]]

    A2 = A[good_idx[1],good_idx[3]]
    # e2 = e[good_idx[1]]
    M2 = M[good_idx[1],good_idx[3]]

    term1 = A1+M1
    term2 = A2-M2

    scaleSq = (1.+term1)*(1.+term2)

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

def chi2LBinsModel(x, scaleSq, scaleSqErr, binCenters1, binCenters2, good_idx):
    
    x = x.reshape((24,5,2))
    
    A = x[...,0]
    # e = x[...,1]
    M = x[...,1]

    scaleSqModelZ = scaleSqFromModelParsSingleMu(A,M, binCenters1, binCenters2, good_idx)

    diffZ = scaleSqModelZ-scaleSqZ

    #batched column vectors
    diffcolZ = np.expand_dims(diffZ,-1)
    #batched row vectors
    diffcolTZ = np.expand_dims(diffZ,-2)

    #batched matrix multiplication
    hScaleSqZ = np.diag(np.reciprocal(np.square(scaleSqErrZ)))
    lbinsZ = 0.5*np.matmul(diffcolTZ,np.matmul(hScaleSqZ, diffcolZ))
    
    return np.sum(lbinsZ)

fileJPsiMC = h5py.File('fitsJMC_corr.hdf5', mode='r')
fileZMC = h5py.File('fitsZMC_corr.hdf5', mode='r')

scaleSqJ = fileJPsiMC['scaleSq'][:]
scaleSqErrJ = fileJPsiMC['scaleSqErr'][:]
good_idxJ = fileJPsiMC['good_idx'][:]

# etas = fileJPsiMC['etas'][:]
etas = np.linspace(-2.4,2.4,25)
etasRed = np.array([-2.4,2.4])
etasC = (etas[:-1] + etas[1:]) / 2

ptsJ =np.array([ 1.1,3.85838168,5.2,6.25981158,7.95232633, 25.        ])
ptsJC = (ptsJ[:-1] + ptsJ[1:]) / 2
nPtBinsJ = ptsJ.shape[0]-1

scaleSqZ_mc = fileZMC['scaleSq'][:]
scaleSqErrZ_mc = fileZMC['scaleSqErr'][:]
good_idxZ = fileZMC['good_idx'][:]

fileZData = h5py.File('fitsZDATA_corr.hdf5', mode='r')
scaleSqZ_data = fileZData['scaleSq'][:]
scaleSqErrZ_data = fileZData['scaleSqErr'][:]
good_idxZ_data = fileZData['good_idx'][:]

scaleSqZ = scaleSqZ_data/scaleSqZ_mc
scaleSqErrZ = np.hypot(scaleSqErrZ_mc/scaleSqZ_mc,scaleSqErrZ_data/scaleSqZ_data)

# scaleSqZ = scaleSqZ_mc
# scaleSqErrZ = scaleSqErrZ_mc

# ptsZ = np.array([-3.14159265, -2.35619449, -1.57079633, -0.78539816,  0., 0.78539816,  1.57079633,  2.35619449,  3.14159265])
ptsZ = np.array([30.,35., 39., 43, 48., 70.])
ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
nPtBinsZ = ptsZ.shape[0]-1

ptsZRed = ptsZ
nPtBinsZRed = ptsZRed.shape[0]-1

nEtaBins = etas.shape[0]-1
nEtaBinsRed =  etasRed.shape[0]-1

nBinsJ = scaleSqJ.shape[0]
nBinsZ = scaleSqZ.shape[0]

nChargeBins=2

# # fitting J
# x=np.ones((nEtaBins,nPtBinsJ,nChargeBins))
# chi2 = chi2LBins(x, scaleSqJ, scaleSqErrJ, good_idxJ)
# print(chi2)

# xmodelJ = pmin(chi2LBins, x.flatten(), args=(scaleSqJ, scaleSqErrJ, good_idxJ), doParallel=False)
# xmodelJ = xmodelJ.reshape((nEtaBins,nPtBinsJ,nChargeBins))

# fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
# hchi2 = jax.jit(jax.hessian(chi2LBins))

# chi2,chi2grad = fgchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)
# chi2hess = hchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)

# hmodel = chi2hess
# covmodel = np.linalg.inv(chi2hess)
# invhess = covmodel

# valmodel,gradmodel = fgchi2(xmodelJ.flatten(), scaleSqJ, scaleSqErrJ, good_idxJ)

# ndof = 2*(nBinsJ) - nEtaBins
# edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

# print("nEtaBins", nEtaBins)
# print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

# errsmodelJ = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsJ,nChargeBins))
# print(xmodelJ, '+/-', errsmodelJ)

## fitting Z

# x=np.ones((nEtaBinsRed,nPtBinsZRed,nChargeBins))
# chi2 = chi2LBins(x, scaleSqZ, scaleSqErrZ, good_idxZ)

# print(chi2)

# xmodelZ = pmin(chi2LBins, x.flatten(), args=(scaleSqZ, scaleSqErrZ, good_idxZ), doParallel=False)
# xmodelZ = xmodelZ.reshape((nEtaBinsRed,nPtBinsZRed,nChargeBins))

# fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
# hchi2 = jax.jit(jax.hessian(chi2LBins))

# chi2,chi2grad = fgchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)
# chi2hess = hchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)

# print(np.linalg.eigh(chi2hess)[0])
# print(np.linalg.eigh(chi2hess)[1][0])
# print(np.all(np.linalg.eigvals(chi2hess) > 0))

# hmodel = chi2hess
# covmodel = np.linalg.inv(chi2hess)
# invhess = covmodel

# valmodel,gradmodel = fgchi2(xmodelZ.flatten(), scaleSqZ, scaleSqErrZ, good_idxZ)

# ndof = nBinsZ - nEtaBinsRed*nPtBinsZRed*nChargeBins
# edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

# print("nBinsZ", nBinsZ, ndof)
# print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

# errsmodelZ = np.sqrt(np.diag(covmodel)).reshape((nEtaBinsRed,nPtBinsZRed,nChargeBins))
# print(xmodelZ, '+/-', errsmodelZ)

# diag = np.diag(np.sqrt(np.diag(invhess)))
# diag = np.linalg.inv(diag)
# corr = np.dot(diag,invhess).dot(diag)

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

# plt.clf()
# plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
# plt.colorbar()
# plt.savefig("corrmatrix.pdf")

# # scaleJ = xmodelJ
# # scaleJError = errsmodelJ

# scaleZ = xmodelZ
# scaleZError = errsmodelZ


fileMC = open('calInputZMC_24etaBins_1ptBins_corr.pkl',"rb")
pkg = pickle.load(fileMC)
binCenters1 = pkg['binCenters1']
binCenters2 = pkg['binCenters2']
binCenters1 = binCenters1[good_idxZ]
binCenters2 = binCenters2[good_idxZ]

### parameters fit

nModelParms = 2

A = np.zeros((nEtaBins, nPtBinsZ),dtype=np.float64)
e = np.zeros((nEtaBins),dtype=np.float64)
M = np.zeros((nEtaBins, nPtBinsZ),dtype=np.float64)

xmodel = np.stack((A,M),axis=-1)
chi2 = chi2LBinsModel(xmodel.flatten(), scaleSqZ, scaleSqErrZ, binCenters1, binCenters2, good_idxZ)

print(chi2)

xmodel = pmin(chi2LBinsModel, xmodel.flatten(), args=(scaleSqZ, scaleSqErrZ, binCenters1, binCenters2, good_idxZ), doParallel=False)
xmodel = xmodel.reshape((24,5,2))

fgchi2 = jax.jit(jax.value_and_grad(chi2LBinsModel))
hchi2 = jax.jit(jax.hessian(chi2LBinsModel))

chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqZ, scaleSqErrZ, binCenters1, binCenters2, good_idxZ)
chi2hess = hchi2(xmodel.flatten(), scaleSqZ, scaleSqErrZ, binCenters1, binCenters2, good_idxZ)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqZ, scaleSqErrZ, binCenters1, binCenters2, good_idxZ)

ndof = 2*(nBinsZ) - nEtaBins*nModelParms
edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

print("chi2/dof = %f/%d = %f" % (2*valmodel,ndof,2*valmodel/float(ndof)))

errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nPtBinsZ, nModelParms))

A = xmodel[...,0]
# e = xmodel[...,1]
M = xmodel[...,1]

# scaleSqModelJ = scaleSqFromModelParsSingleMu(a,b,c,d, binCentersJ, good_idxJ)
# scaleModelJ = np.sqrt(scaleSqModelJ).reshape(nEtaBins,nPtBinsJ)

# scaleSqModelZ = scaleSqFromModelParsSingleMu(a,b,c,d, binCentersZ, good_idxZ)
# scaleModelZ = np.sqrt(scaleSqModelZ).reshape(nEtaBins,nPtBinsZ)

# scalejac = jax.jit(jax.jacfwd(scaleSqFromModelParsSingleMu))(a,b,c,d, binCentersJ, good_idxJ)
# print(scaleSqModelJ.shape,scalejac.shape,covmodel.shape)
# # scalejac = np.reshape(scalejac, (-1,covmodel.shape[0]))
# print(scalejac.shape)
# covscaleModel = np.matmul(scalejac,np.matmul(covmodel,scalejac.T))
# scaleErrsModelJ = np.sqrt(np.diag(covscaleModel))
# scaleErrsModelJ = scaleErrsModelJ.reshape(nEtaBins,nPtBinsJ)

# scalejac = jax.jit(jax.jacfwd(scaleSqFromModelParsSingleMu))(a,b,c,d, binCentersZ, good_idxZ)
# scalejac = np.reshape(scalejac, (-1,covmodel.shape[0]))
# covscaleModel = np.matmul(scalejac,np.matmul(covmodel,scalejac.T))
# scaleErrsModelZ = np.sqrt(np.diag(covscaleModel))
# scaleErrsModelZ = scaleErrsModelJ.reshape(nEtaBins,nPtBinsZ)

print(xmodel, "+/-", errsmodel)
print(edm, "edm")

print(xmodel[...,1],"+/-",errsmodel[:,1])

# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # plt.errorbar(etasC, M[:,4], yerr=errsmodel[:,4,2])
# hep.hist2dplot(A,etas,ptsZ)
# plt.tight_layout()
# plt.show()


etas = onp.array(etas.tolist())
ptsZ = onp.array(ptsZ.tolist())
hM = ROOT.TH2D("MZ","MZ", nEtaBins,etas,nPtBinsZ,ptsZ)
hM = array2hist(M, hM, errsmodel[...,1])
hA = ROOT.TH2D("AZ","AZ", nEtaBins,etas,nPtBinsZ,ptsZ)
hA = array2hist(A, hA, errsmodel[...,0])
fOut= ROOT.TFile("calibrationAlignment_after.root","recreate")
fOut.cd()
hM.Write()
hA.Write()
fOut.Close()

diag = np.diag(np.sqrt(np.diag(invhess)))
diag = np.linalg.inv(diag)
corr = np.dot(diag,invhess).dot(diag)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.clf()
plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("corrmatrix.pdf")

# print("computing scales and errors:")

# aerr = errsmodel[:,0]
# berr = errsmodel[:,1]
# cerr = errsmodel[:,2]
# derr = errsmodel[:,3]

# for i in range(nEtaBins):
#     fig, (ax1) = plt.subplots()
#     # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
#     hep.cms.text('work in progress', ax=ax1)
#     ax1.text(0.95, 0.95,'eta {}'.format(etasC[i]),verticalalignment='top', horizontalalignment='right',
#         transform=ax1.transAxes,
#         color='black', fontsize24)
#     # ax1.set_title("scale", fontsize=18)
#     # ax1.text(0.95, 0.95, 'a: {:.5f}+/-{:.6f}\n b: {:.5f}+/-{:.6f}\n c: {:24f}+/-{:.11f}\n d: {:.1f}+/-{:.1f}\n'.format(a[i],aerr[i],b[i],berr[i],c[i],cerr[i],d[i],derr[i]),verticalalignment='top', horizontalalignment='right',transform=ax1.transAxes,color='black', fontsize24)
#     ax1.errorbar(binCentersJfull[i,2:,1],scaleJ[i,2:], yerr=scaleJError[i,2:], marker="v", label = '$J/\psi$ mc',fmt='v')
#     ax1.errorbar(binCentersZfull[i,...,1],scaleZ[i,...],yerr=scaleZError[i,...], marker="v", label = '$Z$ mc',fmt='v')
#     # ax1.plot(binCentersJfull[i,...,1],scaleModelJ[i,...], color="red")
#     # ax1.fill_between(binCentersJfull[i,...,1],scaleModelJ[i,...]-scaleErrsModelJ[i,...],scaleModelJ[i,...]+scaleErrsModelJ[i,...], color="red", alpha=0.5)
#     # ax1.plot(binCentersZfull[i,...,1],scaleModelZ[i,...], color="red")
#     # ax1.fill_between(binCentersZfull[i,...,1],scaleModelZ[i,...]-scaleErrsModelZ[i,...],scaleModelZ[i,...]+scaleErrsModelZ[i,...], color="red", alpha=0.5)
#     ax1.legend(loc='upper left', frameon=False)
#     plt.tight_layout()
#     plt.savefig('PlotsScale/scale_eta{}.png'.format(round(etasC[i],2)))

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
fileOut = "scale2D_mc_q.hdf5"

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