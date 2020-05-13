import jax.numpy as np
import jax
from jax import grad, hessian, jacobian, config, random
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, fill_hist

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools


def kernelpdf(scale, sigma, datasetGen, masses):

    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    #massWidth = massWidth[np.newaxis,:]
    massWidth = np.reshape(massWidth, len(scale.shape)*(1,) + (-1,))
    
    valsReco = np.reshape(valsMass, len(scale.shape)*(1,) + (-1,1))
    valsGen = np.reshape(valsMass, len(scale.shape)*(1,) + (1,-1))
    
    #valsReco = valsMass[np.newaxis,:,np.newaxis]
    #valsGen = valsMass[np.newaxis,np.newaxis,:]
    
    #scale_ext = scale[:,np.newaxis,np.newaxis]
    #sigma_ext = valsGen*sigma[:,np.newaxis,np.newaxis]
    
    scale_ext = np.reshape(scale, scale.shape + (1,1))
    sigma_ext = valsGen*np.reshape(sigma, sigma.shape + (1,1))

    h = scale_ext*valsGen

    datasetGen_ext = np.expand_dims(datasetGen,-2)

    #analytic integral
    #xscale = np.sqrt(2.)*sigma_ext

    #maxZ = (masses[-1]-h)/xscale
    #minZ = (masses[0]-h)/xscale
    
    #normarg = 0.5*(erf(maxZ)-erf(minZ))
    #I = datasetGen[:,np.newaxis,:]*normarg
    #I = np.sum(I, axis=-1)

    #contribution from each gen mass bin with correct relative normalization
    pdfarg = datasetGen_ext*np.exp(-np.power(valsReco  -h, 2.)/(2 * np.power(sigma_ext, 2.)))/sigma_ext/np.sqrt(2.*np.pi)
    #sum over gen mass bins
    pdf = np.sum(pdfarg,axis=-1)
    #numerical integration over reco mass bins
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)
    
    #print("kernelpdf debug")
    #print(np.any(np.isnan(pdfarg)), np.any(np.isnan(I)))
    pdf = pdf/np.where(pdf>0., I, 1.)

    return pdf

def exppdf(slope, masses):
    
    nBinsMass = masses.shape[0]
    
    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    #massWidth = massWidth[np.newaxis,:]
    massWidth = np.reshape(massWidth, len(slope.shape)*(1,) + (-1,))
    
    
    valsReco = np.reshape(valsMass, len(slope.shape)*(1,) + (-1,))
    
    slope_ext = np.expand_dims(slope,-1)/(masses[-1]-masses[0])
    
    #analytic integral
    #I = (np.exp(-slope_ext*masses[0]) - np.exp(-slope_ext*masses[-1]))/slope_ext
    #print(slope_ext.shape)
    #print(I.shape)
    #I = I[:,np.newaxis]
    
    #pdf = np.exp(-slope_ext*valsReco)
    pdf = np.exp(-slope_ext*(valsReco-masses[0]))
    #numerical integration over reco mass bins
    #I = np.sum(pdf, axis=-1, keepdims=True)
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)
    pdf = pdf/np.where(pdf>0.,I,1.)

    #print(I)
    #print(Ia)
    #print(slope_ext)
    #print(pdf[0])

    return pdf

def gaussianpdf(scale, sigma, masses):

    nBinsMass = masses.shape[0]
    
    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    #massWidth = massWidth[np.newaxis,:]
    massWidth = np.reshape(massWidth, len(scale.shape)*(1,) + (-1,))

    #scale = np.reshape(scale, (-1,) + len(scale.shape)*(1,))
    #sigma = np.reshape(sigma, (-1,) + len(sigma.shape)*(1,))

    valsReco = np.reshape(valsMass, len(scale.shape)*(1,) + (-1,))
    scale_ext = np.expand_dims(scale,-1)
    sigma_ext = np.expand_dims(sigma,-1)

    #print scale_ext.shape, 'scale pdf'
    #print sigma_ext.shape, 'sigma pdf'
    #print valsReco.shape, 'valsReco pdf'

    pdf = np.exp(-0.5*np.square((valsReco-scale_ext)/sigma_ext))
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)
    pdf = pdf/np.where(pdf>0.,I,1.)

    #print pdf.shape, "pdf"

    return pdf
    
def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    return L0/L
    
def scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx, linearize=False):
    
    coeffe1 = binCenters1[...,1]
    coeffM1 = binCenters1[...,0]

    coeffe2 = binCenters2[...,1]
    coeffM2 = binCenters2[...,0]

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0]]
    e1 = e[good_idx[0]]
    M1 = M[good_idx[0]]

    A2 = A[good_idx[1]]
    e2 = e[good_idx[1]]
    M2 = M[good_idx[1]]

    term1 = A1-e1*coeffe1+M1*coeffM1
    term2 = A2-e2*coeffe2-M2*coeffM2
    
    if linearize:
        #neglect quadratic term1*term2
        scaleSq = 1.+term1+term2
    else:
        scaleSq = (1.+term1)*(1.+term2)
        
    return scaleSq

def scaleSqFromModelParsSingleMu(A, e, M, W, etas, binCenters1, good_idx):
    
    coeffe1 = binCenters1[...,1]
    coeffM1 = binCenters1[...,0]
    
    #term1 = A[good_idx[0]]-e[good_idx[0]]*coeffe1+M[good_idx[0]]*coeffM1 + W[good_idx[0]]*np.abs(coeffM1)
    term1 = (A[good_idx[0]]-1.)+np.reciprocal(1+e[good_idx[0]]*coeffe1)+M[good_idx[0]]*coeffM1 + W[good_idx[0]]*np.abs(coeffM1)
    
    scaleSq = np.square(1.-term1)
        
    return scaleSq

def sigmaSqFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx):
    
    #compute sigma from physics parameters

    a1 = a[good_idx[0]]
    b1 = b[good_idx[0]]
    c1 = c[good_idx[0]]
    d1 = d[good_idx[0]]

    a2 = a[good_idx[1]]
    b2 = b[good_idx[1]]
    c2 = c[good_idx[1]]
    d2 = d[good_idx[1]]
    
    ptsq1 = binCenters1[...,2]
    Lsq1 = binCenters1[...,3]
    corr1 = binCenters1[...,4]
    
    ptsq2 = binCenters2[...,2]
    Lsq2 = binCenters2[...,3]
    corr2 = binCenters2[...,4]
    
    res1 = a1*Lsq1 + c1*ptsq1*np.square(Lsq1) + corr1
    res2 = a2*Lsq2 + c2*ptsq2*np.square(Lsq2) + corr2

    sigmaSq = 0.25*(res1+res2)
    
    return sigmaSq

def sigmaSqFromModelParsSingleMu(a,b,c,d, etas, binCenters1, good_idx):
    
    #compute sigma from physics parameters

    pt2 = binCenters1[...,2]
    L2 = binCenters1[...,3]
    #corr = binCenters1[...,4]
    invpt2 = binCenters1[...,4]
    
    sigmaSq = a[good_idx[0]]*L2 + c[good_idx[0]]*pt2*np.square(L2) + b[good_idx[0]]*L2*np.reciprocal(1+d[good_idx[0]]*invpt2/L2)
    #sigmaSq = a[good_idx[0]]*L2 + c[good_idx[0]]*pt2*np.square(L2) + corr

    return sigmaSq

def chi2LBins(x, binScaleSq, binSigmaSq, hScaleSqSigmaSq, etas,*args):
#def chi2LBins(x, binScaleSq, binSigmaSq, hScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx):
    # return the gaussian likelihood (ie 0.5*chi2) from the scales and sigmas squared computed from the
    # physics model parameters vs the values and covariance matrix from the binned fit
    
    A,e,M,W,a,b,c,d = modelParsFromParVector(x)

    if len(args)==3:

        binCenters1=args[0]
        binCenters2=args[1]
        good_idx=args[2]

        scaleSqModel = scaleSqFromModelPars(A,e,M,W,etas, binCenters1, binCenters2, good_idx, linearize=False)
        sigmaSqModel = sigmaSqFromModelPars(a,b,c,d,etas, binCenters1, binCenters2, good_idx)

    else:

        binCenters1=args[0]
        good_idx=args[1]
        scaleSqModel = scaleSqFromModelParsSingleMu(A,e,M,W,etas, binCenters1, good_idx)
        sigmaSqModel = sigmaSqFromModelParsSingleMu(a,b,c,d,etas, binCenters1, good_idx)
    
    scaleSqSigmaSqModel = np.stack((scaleSqModel,sigmaSqModel), axis=-1)
    scaleSqSigmaSqBinned = np.stack((binScaleSq,binSigmaSq), axis=-1)
    
    #print(scaleSqSigmaSqModel, "scaleSqSigmaSqModel")
    #print(scaleSqSigmaSqBinned,"scaleSqSigmaSqBinned")
    
    diff = scaleSqSigmaSqModel-scaleSqSigmaSqBinned
    print("diff.shape")
    print(diff.shape)
    
    #batched column vectors
    diffcol = np.expand_dims(diff,-1)
    
    #batched row vectors
    diffcolT = np.expand_dims(diff,-2)

    print("chi2 shapes")
    print(diffcol.shape, diffcolT.shape, hScaleSqSigmaSq.shape)

    #batched matrix multiplication
    lbins = 0.5*np.matmul(diffcolT,np.matmul(hScaleSqSigmaSq, diffcol))
    
    return np.sum(lbins)
    
    #print(chi2bins.shape)
    
    #return chi2 
    
    
def chi2SumBins(x, binScaleSq, binSigmaSq, covScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx):
    chi2Binspartial = functools.partial(chi2Bins, etas=etas, binCenters1=binCenters1, binCenters2=binCenters2,good_idx=good_idx)
    chi2bins = jax.vmap(chi2Binspartial(x, binScaleSq, binSigmaSq, covScaleSqSigmaSq))
    chi2 = np.sum(chi2bins)
    return chi2
                        

def modelParsFromParVector(x):
    x = x.reshape((-1,8))
    
    A = x[...,0]
    e = x[...,1]
    M = x[...,2]
    W = x[...,3]
    a = x[...,4]
    c = x[...,5]
    b = x[...,6]
    d = x[...,7]

    return A,e,M,W,a,b,c,d

def scaleSigmaFromModelParVector(x, etas, binCenters1, binCenters2, good_idx):
    A,e,M,a,b,c,d = modelParsFromParVector(x)
    return scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx)

def scaleFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx):
    scaleSq = scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx, linearize=False)
    return np.sqrt(scaleSq)

def sigmaFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx):
    
    sigmaSq = sigmaSqFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx)
    return np.sqrt(sigmaSq)

def scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx):

    scale = scaleFromModelPars(A,e,M,etas, binCenters1, binCenters2, good_idx)
    sigma = sigmaFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx)
    
    return scale,sigma

def splitTransformPars(x, ndata, nEtaBins, nBins, isJ=True):
    #A = x[:nEtaBins]
    A = 0.01*np.tanh(x[:nEtaBins])

    if isJ:
        #e = x[nEtaBins:2*nEtaBins]
        #M = x[2*nEtaBins:3*nEtaBins]
        #a = np.exp(x[3*nEtaBins:4*nEtaBins])
        #nsig = x[4*nEtaBins:4*nEtaBins+nBins]
        #nbkg = x[4*nEtaBins+nBins:]
        
        e = 0.01*np.tanh(x[nEtaBins:2*nEtaBins])
        M = 0.01*np.tanh(x[2*nEtaBins:3*nEtaBins])
        a = 1e-6 + 0.07e-3*np.exp(x[3*nEtaBins:4*nEtaBins])
        #a = 0.07e-3*np.exp(x[3*nEtaBins:4*nEtaBins])
        nsig = ndata*np.exp(x[4*nEtaBins:4*nEtaBins+nBins])
        nbkg = ndata*np.square(x[4*nEtaBins+nBins:4*nEtaBins+2*nBins])
        slope = x[4*nEtaBins+2*nBins:]

    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        a = x[2*nEtaBins:3*nEtaBins]
        nsig = np.exp(x[3*nEtaBins:])

    b = 0.03e-3*np.ones((nEtaBins))
    c = 15.e-9*np.ones((nEtaBins))
    d = 370.*np.ones((nEtaBins))
    
    return A,e,M,a,b,c,d,nsig,nbkg,slope
    

def scaleSqSigmaSqFromBinsPars(x):
    scale, sigma = scaleSigmaFromBinPars(x)
    return np.square(scale), np.square(sigma)

def scaleSigmaFromBinPars(x):
    #flexible on shape of input array as long as last dimension indexes the parameters within a bin
    scale = x[...,0]
    sigma = x[...,1]
    
    #parameter transformation for bounds
    #(since the bounds are reached only asymptotically, the fit will not converge well
    #if any of the parameters actually lie outside this region, these are just designed to protect
    #against pathological phase space during minimization)
    scale = 1. + 1e-1*np.tanh(scale)
    #sigma = 5e-3*np.exp(2.*np.tanh(sigma))
    sigma = 0.01*np.exp(3.*np.tanh(sigma))
    
    return scale, sigma
    
def bkgModelFromBinPars(x):
    #flexible on shape of input array as long as last dimension indexes the parameters within a bin
    fbkg = x[...,2]
    slope = x[...,3]
    
    # Transformation with hard bounds like in Minuit
    # Fit should still behave ok if bound is reached, though uncertainty
    # associated with this parameter will be underestimated
    fbkg = 0.5*(1.+np.sin(fbkg))
    
    return fbkg, slope
    
def nllBinsFromBinPars(x, dataset, datasetGen, masses):
    # For fitting with floating signal and background parameters
    
    scale, sigma = scaleSigmaFromBinPars(x)
    fbkg, slope = bkgModelFromBinPars(x)
    
    return nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses)

def nllBinsFromSignalBinPars(x, fbkg, slope, dataset, datasetGen, masses):
    # For fitting with fixed background parameters
    scale, sigma = scaleSigmaFromBinPars(x)
    
    return nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses)

def nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses):
    sigpdf = kernelpdf(scale,sigma, datasetGen, masses)
    bkgpdf = exppdf(slope, masses)
    
    fbkg_ext = np.expand_dims(fbkg,-1)
    
    pdf = (1.-fbkg_ext)*sigpdf + fbkg_ext*bkgpdf
    
    nll = np.sum(-dataset*np.log(np.where(dataset>0., pdf, 1.)),axis=-1)
    
    #constraint on slope to keep fit well behaved when fbkg->0
    slopesigma = 5.
    nll += 0.5*np.square(slope)/slopesigma/slopesigma
    return nll

def nllBinsFromBinParsRes(x, dataset, masses):
    # For fitting with fixed background parameters
    scale, sigma = scaleSigmaFromBinPars(x)

    #print scale,sigma
    
    return nllBinsResolution(scale, sigma, dataset, masses)

def nllBinsResolution(scale, sigma, dataset, masses):
    
    pdf = gaussianpdf(scale, sigma, masses)
    #print dataset.shape, 'dataset.shape'
    #print scale.shape, 'scale.shape'
    #print pdf.shape, 'pdf.shape'
    nll = np.sum(-dataset*np.log(np.where(dataset>0., pdf, 1.)),axis=-1)

    #print nll.shape, 'nll.shape'
    
    return nll

def plotsBkg(scale,sigma,fbkg,slope,dataset,datasetGen,masses,isJ,etas, binCenters1, binCenters2, good_idx):


    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)    
    minR = masses[0]
    maxR = masses[-1]
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    
    masseslow = masses[:-1]
    
    nsig = (1.-fbkg)*ndata
    nbkg = fbkg*ndata
    
    sigpdf = nsig[:,np.newaxis]*massWidth*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,np.newaxis]*massWidth*exppdf(slope,masses)

    pdf = sigpdf+bkgpdf


    #for ibin in range(nBins):
    for ibin in range(0,nBins,10):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]        
        
        scale_bin = scale[ibin]
        sigma_bin = sigma[ibin]
        fbkg_bin = fbkg[ibin]
        slope_bin = slope[ibin]
        n_true_bin = ndata[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(masseslow, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n fbkg: {:.3f}\n slope: {:.3f}\n ntrue: {:.0f}'\
                        .format(scale_bin,sigma_bin,fbkg_bin,slope_bin,n_true_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
        ax1.set_xlim([minR, maxR])
        
        ax1.plot(masseslow, pdf[ibin,:])
        ax1.plot(masseslow, bkgpdf[ibin,:], ls='--')
                
        ax2.errorbar(masseslow,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('dimuon mass')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])


        plt.savefig('PLOTS{}/plot_{}{}{}{}.png'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
        plt.close(fig)

def plotsSingleMu(scale,sigma,dataset,masses):


    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)
    print ndata.shape    
    minR = masses[0]
    maxR = masses[-1]
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    
    masseslow = masses[:-1]
    
    pdf = ndata[:,np.newaxis]*massWidth*gaussianpdf(scale, sigma, masses)
    print gaussianpdf(scale, sigma, masses)


    #for ibin in range(nBins):
    for ibin in range(0,nBins,10):
        
        scale_bin = scale[ibin]
        sigma_bin = sigma[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(masseslow, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n'\
                        .format(scale_bin,sigma_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
        ax1.set_xlim([minR, maxR])
        
        ax1.plot(masseslow, pdf[ibin,:])
                
        ax2.errorbar(masseslow,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('p_rec/p_gen')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])


        plt.savefig('PLOTSMCTruth/plot_{}.png'.format(ibin))
        plt.close(fig)

