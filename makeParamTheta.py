import math
import h5py
import pickle
import lz4.frame
from array import array
import ROOT
import jax
import jax.numpy as np
import numpy as onp
import scipy
from scipy.interpolate import make_interp_spline, BSpline
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)
from jax import random
import numpy as onp
from obsminimization import pmin

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])

ROOT.gROOT.SetBatch(True)

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

def TH1F2np(histo, overflow=False):
    content = histo.GetArray()
    content.SetSize(histo.GetNbinsX() + 2)
    if overflow:
        content = onp.array(content)[1:]
    else:
        content = onp.array(content)[1:-1]
    binning = onp.array(histo.GetXaxis().GetXbins())
    if overflow:
        binning = onp.append(binning, [onp.inf])
        
    return content, binning

def TH2F2np(histo, overflow=False):
    binningx = onp.array(histo.GetXaxis().GetXbins())
    binningy = onp.array(histo.GetYaxis().GetXbins())
    if overflow:
        binningx = onp.append(binningx, [onp.inf])
        binningy = onp.append(binningy, [onp.inf])
    if overflow:
        content = onp.zeros((histo.GetNbinsX() + 1, histo.GetNbinsY() + 1))
    else:
        content = onp.zeros((histo.GetNbinsX(), histo.GetNbinsY()))
    for xbin in range(1, histo.GetNbinsX() + (2 if overflow else 1)):
        for ybin in range(1, histo.GetNbinsY() + (2 if overflow else 1)):
            content[xbin - 1, ybin - 1] = histo.GetBinContent(xbin, ybin)
    return content, binningx, binningy

def logsigpdfbinned(mu,sigma,krs):
    
    krs = krs[np.newaxis,np.newaxis,np.newaxis,:]
    width = krs[...,1:] - krs[...,:-1]
    #krl = krs[...,0]
    #krh = krs[...,-1]
    
    #krl = krl[...,np.newaxis]
    #krh = krh[...,np.newaxis]
    
    kr = 0.5*(krs[...,1:] + krs[...,:-1])

    #sigma = 2e-3
    
    alpha = 3.0
    # alpha = 2.0
    #alpha = 1.0
    # alpha = 5.0
    #alpha = 100.0
    alpha1 = alpha
    alpha2 = alpha
    
    A1 = np.exp(0.5*alpha1**2)
    A2 = np.exp(0.5*alpha2**2)
    
    t = (kr - mu)/sigma
    tleft = np.minimum(t,-alpha1)
    tright = np.maximum(t,alpha2)
    tcore = np.clip(t,-alpha1,alpha2)
    #tleft = np.where(t<-alpha1, t, -alpha1)
    #tright = np.where(t>=alpha2, t, alpha2)
    
    #pdfcore = np.exp(-0.5*tcore**2)
    #pdfleft = A1*np.exp(alpha1*tleft)
    #pdfright = A2*np.exp(-alpha2*tright)
    
    pdfcore = -0.5*tcore**2
    pdfleft = np.log(A1) + alpha1*tleft
    pdfright = np.log(A2) - alpha2*tright
    
    logpdf = np.where(t<-alpha1, pdfleft, np.where(t<alpha2, pdfcore, pdfright))
    
    I = np.sum(width*np.exp(logpdf),axis=-1,keepdims=True)
    
    #Icore = (scipy.special.ndtr(alpha2) - scipy.special.ndtr(-alpha1))*sigma*np.sqrt(2.*np.pi)
    #Ileft = (sigma/alpha1)*np.exp(-0.5*alpha1**2)
    #Iright = (sigma/alpha2)*np.exp(-0.5*alpha2**2)
    
    #I = Icore + Ileft + Iright
    
    #print("I")
    #print(I)
    
    return logpdf - np.log(I)

def nllbinned(parms, dataset, krs):
    mu = parms[...,0]
    sigma = parms[...,1]
    
    #sigma = np.sqrt(sigmasq)
    sigma = np.where(sigma>0.,sigma,np.nan)
    
    #mu = 1. + 0.1*np.tanh(mu)
    #sigma = 1e-3*(1. + np.exp(sigma))
    
    mu = mu[...,np.newaxis]
    sigma = sigma[...,np.newaxis]
    
    logpdf = logsigpdfbinned(mu,sigma,krs)
    #logpdf = loggauspdfbinned(mu,sigma,krs)
    
    nll = -np.sum(dataset*logpdf, axis=-1)
    #nll += np.squeeze(sigma,axis=-1)**2
    #nll += sigma**2
    
    return nll

# def scale(A,e,M,k,q):
#     print(A.shape,k.shape,q.shape)
#     return A + q*M/k - e*k

# def sigmasq(a, c, k):
#     return a + c/k**2
def scale(M,k,q):
    return M
    #return q*k*A + e*k**2 + M + q*R

def sigmasq(a, c, k):
    #return a + c/k**2
    return a*k**2 + c
    #return c + a*k**2
    
def scalesigma(parms, qs, ks):
    M = parms[..., 0, np.newaxis, np.newaxis]
    a = parms[..., 1, np.newaxis, np.newaxis]
    c = parms[..., 2, np.newaxis, np.newaxis]
    
    qs = qs[np.newaxis, :, np.newaxis]
    ks = ks[:, np.newaxis, :]
    
    scaleout = scaleout = np.ones_like(qs)*np.ones_like(ks)*scale(M,ks,qs)
    sigmasqout = sigmasq(a,c,ks)
    sigmaout = np.sqrt(sigmasqout)
    sigmaout = np.ones_like(qs)*sigmaout
    
    return np.stack([scaleout, sigmaout], axis=-1)

jacscalesigma = jax.jit(jax.jacfwd(lambda *args: scalesigma(*args).flatten()))

def nllbinnedmodel(parms, dataset, qs, ks, krs):
    #A = parms[..., 0, np.newaxis, np.newaxis, np.newaxis]
    #e = parms[..., 1, np.newaxis, np.newaxis, np.newaxis]
    #M = parms[..., 2, np.newaxis, np.newaxis, np.newaxis]
    #R = parms[..., 3, np.newaxis, np.newaxis, np.newaxis]
    #a = parms[..., 4, np.newaxis, np.newaxis, np.newaxis]
    #c = parms[..., 5, np.newaxis, np.newaxis, np.newaxis]
    
    #qs = qs[np.newaxis, :, np.newaxis, np.newaxis]
    #ks = ks[np.newaxis, np.newaxis, :, np.newaxis]

    scalesigmaout = scalesigma(parms, qs, ks)
    mu = scalesigmaout[...,0]
    sigma = scalesigmaout[...,1]
    
    mu = mu[...,np.newaxis]
    sigma = sigma[...,np.newaxis]
    
    #mu = scale(A,e,M,R,ks,qs)
    #sigsq = sigmasq(a,c,ks)
    #sigma = np.sqrt(sigsq)
    
    logpdf = logsigpdfbinned(mu,sigma,krs)
    
    nll = -np.sum(dataset*logpdf, axis=(-1,-2,-3))
    
    #e = parms[..., 1]
    #sige = 1e-6
    #nll += np.sum(np.square(e), axis=-1)/sige/sige
    
    
    return nll

def plotsSingleMu(scale,sigma,dataset,masses):

    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)
    print(ndata.shape)    
    minR = masses[0]
    maxR = masses[-1]
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    
    masseslow = masses[:-1]
    print(logsigpdfbinned(scale, sigma, masses).shape)
    pdf = np.squeeze(ndata[:,np.newaxis]*massWidth*np.exp(logsigpdfbinned(scale, sigma, masses)))
    print(pdf)
    for ibin in range(nBins):
        
        scale_bin = np.squeeze(scale[ibin])
        sigma_bin = np.squeeze(sigma[ibin])

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(masseslow, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n'.format(scale_bin,sigma_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
        ax1.set_xlim([minR, maxR])
        #ax1.set_yscale('log')
        ax1.plot(masseslow, pdf[ibin,:])
                
        ax2.errorbar(masseslow,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('$(k_{rec}-k_{gen})/k_{gen}$')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])

        print('MCTruthFitZ/plot_{}.png'.format(ibin))
        plt.savefig('MCTruthFitZ/plot_{}.png'.format(ibin))
        plt.close(fig)

def sigmaSqFromModelParsSingleMu(a,b,c,d, binCenters, good_idx):
    
    #compute sigma from physics parameters
    a = a[good_idx[0]]
    b = b[good_idx[0]]
    c = c[good_idx[0]]
    d = d[good_idx[0]]

    pt2 = binCenters[...,1]
    L2 = binCenters[...,0]
    invpt2 = binCenters[...,2]
    
    sigmaSq = a*L2 + c*pt2*np.square(L2) + b*L2*np.reciprocal(1.+d*invpt2/L2)

    return sigmaSq

def sigmaSqFromModelParsSingleMuFromVec(x,binCenters, good_idx):

    a = x[...,0]
    b = x[...,1]
    c = x[...,2]
    d = x[...,3]

    return sigmaSqFromModelParsSingleMu(a,b,c,d, binCenters, good_idx)

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

nEtaBins = 24
etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
etasC = (etas[:-1] + etas[1:]) / 2
res=np.linspace(-0.005,0.005,101)
qcs = onp.array([-1.,1.], dtype=np.float64)

fileZMC = 'JPsiInputData/ZMC.pkl.lz4'
with (lz4.frame.open(fileZMC, "r")) as openfile:
    resultdict_mc = pickle.load(openfile)
harrayZpl = resultdict_mc['Z_mc_resplus'].values()
harrayZm = resultdict_mc['Z_mc_resminus'].values()
harrayZplmeans = resultdict_mc['Z_mc_resplusmeans'].values()

ptsZ = resultdict_mc['Z_mc_resplus'].axes[0].edges
ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
nPtBinsZ = ptsZ.shape[0]-1

harrayZ =  harrayZpl+harrayZm

fileJPsiMC = 'JPsiInputData/JPsiMC.pkl.lz4'
with (lz4.frame.open(fileJPsiMC, "r")) as openfile:
    resultdict_mc = pickle.load(openfile)
harraypl = resultdict_mc['Jpsi_mc_resplus'].values()
harraym = resultdict_mc['Jpsi_mc_resminus'].values()
harrayJplmeans = resultdict_mc['Jpsi_mc_resplusmeans'].values()
# harrayJmmeans = fileJPsiMC['Jpsi_distr_mcminus_means'][:]
harrayJ =  harraypl+harraym

ptsJ =resultdict_mc['Jpsi_mc_resplus'].axes[0].edges
ptsJC = (ptsJ[:-1] + ptsJ[1:]) / 2
nPtBinsJ = ptsJ.shape[0]-1

good_idxJ = np.nonzero(np.sum(harraypl,axis=(-1)).T>0.)
good_idxZ = np.nonzero(np.sum(harrayZpl,axis=(-1)).T>0.)

nllbinnedsum = lambda *args: np.sum(nllbinned(*args),axis=(0,1,2))
gbinned = jax.grad(nllbinnedsum)

def fgbinned(*args):
    return nllbinned(*args), gbinned(*args)

gbinnedsum = lambda *args: np.sum(gbinned(*args),axis=(0,1,2))
jacbinned = jax.jacrev(gbinnedsum)
hbinned = lambda *args: np.moveaxis(jacbinned(*args),0,-1)

fgbinned = jax.jit(fgbinned)
hbinned = jax.jit(hbinned)

# fitting J
xmu = np.zeros((nEtaBins,2,nPtBinsJ),dtype=np.float64)
xsigma = 1.*np.ones((nEtaBins,2,nPtBinsJ),dtype=np.float64)
xbinned = np.stack((xmu,xsigma),axis=-1)

harraypl = np.swapaxes(harraypl,0,1)
harraym = np.swapaxes(harraym,0,1)

hdset = np.stack((harraypl,harraym),axis=1)

# good_idxJ = np.nonzero(np.sum(hdset,axis=(-1))>0.)
# plotsSingleMu(xbinned[...,0,np.newaxis][good_idxJ],xbinned[...,1,np.newaxis][good_idxJ],hdset[good_idxJ],res)
# assert(0)

xbinned = pmin(fgbinned, xbinned, (hdset,res), jac=True, h=hbinned, edmtol = 1e-5)

hessbinned = hbinned(xbinned, hdset, res)
covbinned = np.linalg.inv(hessbinned)

errsbinned =  np.sqrt(np.diagonal(covbinned, offset=0, axis1=-1, axis2=-2))

# print(xbinned,"+/-",errsbinned)

# good_idxJ = np.nonzero(np.sum(hdset,axis=(-1))>0.) 
# plotsSingleMu(xbinned[...,0,np.newaxis][good_idxJ],xbinned[...,1,np.newaxis][good_idxJ],hdset[good_idxJ],res)
# assert(0)

xmodelJ = np.square(xbinned[...,0])
errsmodelJ = 2.*xmodelJ*errsbinned[...,0]

xmodelJ = xbinned[...,0]
errsmodelJ = errsbinned[...,0]

# fitting Z
xmu = np.zeros((nEtaBins,2,nPtBinsZ),dtype=np.float64)
xsigma = 0.001*np.ones((nEtaBins,2,nPtBinsZ),dtype=np.float64)
xbinned = np.stack((xmu,xsigma),axis=-1)

harrayZpl = np.swapaxes(harrayZpl,0,1)
harrayZm = np.swapaxes(harrayZm,0,1)
print(harrayZpl.shape)
hdset = np.stack((harrayZpl,harrayZm),axis=1)
print(hdset.shape)

# good_idxZ = np.nonzero(np.sum(hdset,axis=(-1))>0.) 
# plotsSingleMu(xbinned[...,0,np.newaxis][good_idxZ],xbinned[...,1,np.newaxis][good_idxZ],hdset[good_idxZ],res)

xbinned = pmin(fgbinned, xbinned, (hdset,res), jac=True, h=hbinned, edmtol = 1e-5)

hessbinned = hbinned(xbinned, hdset, res)
covbinned = np.linalg.inv(hessbinned)

errsbinned =  np.sqrt(np.diagonal(covbinned, offset=0, axis1=-1, axis2=-2))

# print(xbinned,"+/-",errsbinned)

# good_idxZ = np.nonzero(np.sum(hdset,axis=(-1))>0.) 
# plotsSingleMu(xbinned[...,0,np.newaxis][good_idxZ],xbinned[...,1,np.newaxis][good_idxZ],hdset[good_idxZ],res)

xmodelZ = np.square(xbinned[...,0])
errsmodelZ = 2.*xmodelZ*errsbinned[...,0]

xmodelZ = xbinned[...,0]
errsmodelZ = errsbinned[...,0]

#nBinsJ = xmodelJ.ravel().shape[0]
#nBinsZ = xmodelZ.ravel().shape[0]

#xmodelJ = xmodelJ
#xmodelZ = xmodelZ

#errsmodelJ = np.array(errsmodelJ).reshape(nEtaBins,2,nPtBinsJ)
#errsmodelZ = np.array(errsmodelZ).reshape(nEtaBins,2,nPtBinsZ)

#bincenter_idx = [0,6,8]
harraypl = np.swapaxes(harraypl,0,1)
binCentersJfull = np.swapaxes((harrayJplmeans/np.expand_dims(np.sum(harraypl,axis=-1),axis=-1)),0,1)
harrayZpl = np.swapaxes(harrayZpl,0,1)
binCentersZfull = np.swapaxes((harrayZplmeans/np.expand_dims(np.sum(harrayZpl,axis=-1),axis=-1)),0,1)

# parameters fit

harraypl = np.swapaxes(harraypl,0,1)
# harraym = np.swapaxes(harraym,0,1)
harrayZpl = np.swapaxes(harrayZpl,0,1)
# harrayZm = np.swapaxes(harrayZm,0,1)

htotplus = np.concatenate((harraypl,harrayZpl),axis=1)
htotminus = np.concatenate((harraym,harrayZm),axis=1)

# hdset = np.stack((htotminus,htotplus),axis=1)
hdset = np.stack((harraym,harraypl),axis=1)
# hdset = np.stack((harrayZm,harrayZpl),axis=1)

# kcs = np.concatenate((binCentersJfull[...,2],binCentersZfull[...,2]),axis=1)
kcs = binCentersJfull[...,2]
# kcs = binCentersZfull[...,2]

nllbinnedmodelsum = lambda *args: np.sum(nllbinnedmodel(*args),axis=(0,))
gbinnedmodel = jax.grad(nllbinnedmodelsum)

def fgbinnedmodel(*args):
    return nllbinnedmodel(*args), gbinnedmodel(*args)

gbinnedmodelsum = lambda *args: np.sum(gbinnedmodel(*args),axis=(0,))
jacbinnedmodel = jax.jacrev(gbinnedmodelsum)
hbinnedmodel = lambda *args: np.moveaxis(jacbinnedmodel(*args),0,-1)

fgbinnedmodel = jax.jit(fgbinnedmodel)
hbinnedmodel = jax.jit(hbinnedmodel)

parmscale = np.zeros((nEtaBins, 1), dtype=np.float64)
parmsigma0 = 1.*np.ones((nEtaBins, 1), dtype=np.float64)
parmsigma1 = 0.*np.ones((nEtaBins, 1), dtype=np.float64)

parmsmodel = np.concatenate([parmscale, parmsigma0, parmsigma1], axis=-1)

parmsmodel = pmin(fgbinnedmodel, parmsmodel, (hdset,qcs, kcs, res), jac=True, h=hbinnedmodel, edmtol = 1e-5)

x = parmsmodel

hess = hbinnedmodel(parmsmodel, hdset, qcs, kcs, res)
cov = np.linalg.inv(hess)
xerr = np.sqrt(np.diagonal(cov, axis1=-2,axis2=-1))

nModelParms = 3

invhess = onp.zeros(shape=(24*nModelParms,24*nModelParms), dtype=x.dtype)
print("nModelParms", nModelParms)
print("x.shape", x.shape)
print("cov.shape", cov.shape)
for i in range(24):
    invhess[i*nModelParms:(i+1)*nModelParms, i*nModelParms:(i+1)*nModelParms] = cov[i,...]
print(invhess.shape)

covdiag = np.diag(invhess)
corr = invhess/np.sqrt(covdiag[:,np.newaxis]*covdiag[np.newaxis,:])

# print(x,"+/-",xerr)

M = x[...,0]

Merr = xerr[...,0]

# fig, ax1 = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# hep.histplot(A.ravel(), etas, yerr=Aerr.ravel(), histtype="errorbar",label="mc")
# plt.tight_layout()
# plt.show()

etaarr = onp.array(etas.tolist())
hM = ROOT.TH1D("M", "M", nEtaBins, etaarr)

hM = array2hist(-1*M, hM, Merr)


f = ROOT.TFile("outClosureTruthTheta.root", 'recreate')
f.cd()

hM.Write()

correlationHist = ROOT.TH2D('correlation_matrix', 'correlation matrix', 24*nModelParms, 0., 1., 24*nModelParms, 0., 1.)
covarianceHist  = ROOT.TH2D('covariance_matrix', 'covariance matrix', 24*nModelParms, 0., 1., 24*nModelParms, 0., 1.)
correlationHist.GetZaxis().SetRangeUser(-1., 1.)

array2hist(corr, correlationHist,np.zeros_like(corr))
array2hist(invhess, covarianceHist,np.zeros_like(invhess))

correlationHist.Write()
covarianceHist.Write()
assert(0)
# for plot in plots:
#     plot.Write()

# qs = qcs[np.newaxis, :, np.newaxis]
# ks = kcs[:, np.newaxis, :]
# M = M[:, np.newaxis, np.newaxis]
# scaleModel = scale(M,ks,qs)
# scaleModelMinus = scale(M,np.flip(ks,axis=0),qs)

sigmaJ = xmodelJ
sigmaJError = errsmodelJ.reshape(nEtaBins,2,nPtBinsJ)

sigmaZ = xmodelZ
sigmaZError = errsmodelZ.reshape(nEtaBins,2,nPtBinsZ)
# print(sigmaZ.shape,binCentersZfull.shape)

fileOut = "scalemctruth_afterJ.hdf5"

# scale = np.concatenate((sigmaJ,sigmaZ),axis=-1)
# scaleError = np.concatenate((sigmaJError,sigmaZError),axis=-1)
# bins =  np.concatenate((binCentersJfull[...,4],binCentersZfull[...,4]),axis=-1)

# with h5py.File(fileOut, mode="w") as f:
#     dtype = 'float64'
#     dset_scale = f.create_dataset('scale', scale.shape, dtype=dtype)
#     dset_scale[...] = scale
#     dset_scale = f.create_dataset('scaleErr', scaleError.shape, dtype=dtype)
#     dset_scale[...] = scaleError
#     dset_scale = f.create_dataset('bins', bins.shape, dtype=dtype)
#     dset_scale[...] = bins


for i in range(nEtaBins):
    fig, (ax1) = plt.subplots()
    hep.cms.text('work in progress', ax=ax1)
    ax1.text(0.95, 0.95, '$\eta$ {}'.format(round(etasC[i],2)),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=12)
    # ax1.text(0.95, 0.95, 'a: {:.5f}+/-{:.6f}\n b: {:.5f}+/-{:.6f}\n c: {:.10f}+/-{:.11f}\n d: {:.1f}+/-{:.1f}\n'\
    #                     .format(a[i],aerr[i],b[i],berr[i],c[i],cerr[i],d[i],derr[i]),
    #     verticalalignment='top', horizontalalignment='right',
    #     transform=ax1.transAxes,
    #     color='black', fontsize=10)
    binstotplus = np.concatenate((binCentersJfull[i,...,4],binCentersZfull[i,...,4]))
    binstotminus= np.concatenate((-1*binCentersZfull[i,...,4],-1*binCentersJfull[i,...,4]))
    # xnew = np.linspace(binstot.min(), binstot.max(), 100) 
    # modeltot = np.concatenate((sigmaModelJ[i,...][sigmaModelJ[i,...]>0.],sigmaModelZ[i,...]))
    # errmodeltot = np.concatenate((SigmaErrsModelJ[i,...][SigmaErrsModelJ[i,...]>0.],SigmaErrsModelZ[i,...]))
    # spl = make_interp_spline(binstot, modeltot, k=3)  # type: BSpline
    # power_smooth = spl(xnew)
    # splUp = make_interp_spline(binstot, modeltot+errmodeltot, k=3)  # type: BSpline
    # power_smoothUp = spl(xnew)
    # splDown = make_interp_spline(binstot, modeltot-errmodeltot, k=3)  # type: BSpline
    # power_smoothDown = spl(xnew)
    # ax1.errorbar(binCentersJfull[i,...,1],sigmaJ[i,...], yerr=sigmaJError[i,...], marker="v", label = '$J/\psi$ mc')
    # ax1.plot(binstotminus,scaleModelMinus[i,0,...])
    # ax1.plot(binstotplus,scaleModel[i,1,...])
    ax1.errorbar(binCentersJfull[i,...,4],sigmaJ[i,0,...], yerr=sigmaJError[i,0,...], marker=".", label = '$J/\psi$ mc plus', fmt=".")
    ax1.errorbar(-1*binCentersJfull[i,...,4],sigmaJ[i,1,...], yerr=sigmaJError[i,1,...], marker=".", label = '$J/\psi$ mc minus', fmt=".")

    ax1.errorbar(binCentersZfull[i,...,4],sigmaZ[i,0,...], yerr=sigmaZError[i,0,...], marker=".", label = '$Z$ mc plus', fmt=".")
    ax1.errorbar(-1*binCentersZfull[i,...,4],sigmaZ[i,1,...], yerr=sigmaZError[i,1,...], marker=".", label = '$Z$ mc minus', fmt=".")

    # ax1.errorbar(binCentersJfull[i,...,1],sigmaJ[i,1,...], yerr=sigmaJError[i,1,...], marker=".", label = '$J/\psi$ mc minus')
    # ax1.errorbar(binCentersZfull[i,...,4],sigmaZ[i,...], yerr=sigmaZError[i,...], marker=".", label = '$Z$ mc plus', fmt=".")
    # ax1.errorbar(binCentersZfull[i,...,1],sigmaZ[i,1,...], yerr=sigmaZError[i,1,...], marker=".", label = '$Z$ mc minus')
    # ax1.plot(xnew,power_smooth, color="red")
    # ax1.fill_between(xnew,power_smoothDown,power_smoothUp, color="red", alpha=0.5)
    # ax1.set_ylabel('$\sigma_{p_T}/{p_T}$')
    # ax1.fill_between(binstot,-0.0001,0.0001, color="red", alpha=0.2)
    ax1.set_ylabel('scale $\phi$')
    ax1.set_xlabel('$qp_T$')
    ax1.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    print('PlotsScaleTruthAfter/scaletheta_eta{}.png'.format(round(etasC[i],2)))
    plt.savefig('PlotsScaleTruthAfter/scaletheta_eta{}.png'.format(round(etasC[i],2)))
