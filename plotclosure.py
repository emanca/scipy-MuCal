import numpy as np
import h5py
import matplotlib.pyplot as plt
import mplhep as hep

fileMC = h5py.File('scale2D_mc.hdf5', mode='r')
fileDATA = h5py.File('scale2D_data.hdf5', mode='r')

scalemc = fileMC['scale'][:]
scalemcErr = fileMC['scaleErr'][:]

scaledata = fileDATA['scale'][:]
scaledataErr = fileDATA['scaleErr'][:]

scale = scaledata[...] - scalemc[...]
scaleErr = np.hypot(scaledataErr[...],scalemcErr[...])

fileMC_sm = h5py.File('scale2D_mc_gensm.hdf5', mode='r')
# fileMC_sm = h5py.File('scale2DJ_mc.hdf5', mode='r')
scalemc_sm = fileMC_sm['scale'][:]
scalemcErr_sm = fileMC_sm['scaleErr'][:]

# fileDATA_sm = h5py.File('scale2D_data_gensm.hdf5', mode='r')
fileDATA_sm = h5py.File('scale2D_mc_gensm02.hdf5', mode='r')
scaledata_sm = fileDATA_sm['scale'][:]
scaledataErr_sm = fileDATA_sm['scaleErr'][:]

scale_sm = scaledata_sm[...] - scalemc_sm[...]
scaleErr_sm = np.hypot(scaledataErr_sm[...],scalemcErr_sm[...])

etas = np.linspace(-2.4,2.4,25)
etasC = (etas[:-1] + etas[1:]) / 2
ptsZ = np.array([30.,35., 39., 43, 48., 70.])
# ptsZ = np.array([ 2.,3.9,5.2,6.3,8.7,25.])
ptsZC = (ptsZ[:-1] + ptsZ[1:]) / 2
nPtBinsZ = ptsZ.shape[0]-1

print(scalemc,scaledata)

fig, (ax1,ax2) = plt.subplots(2)
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
hep.cms.text('work in progress', ax=ax1)
# hep.histplot(scale.ravel(), ptsZ, yerr=scaleErr.ravel(), histtype="errorbar",label="unsmeared")
hep.histplot(scalemc.ravel(), ptsZ, yerr=scalemcErr.ravel(), histtype="errorbar",label="mc smeared 50%",ax=ax1)
# hep.histplot(scaledata_sm.ravel(), ptsZ, yerr=scaledataErr_sm.ravel(), histtype="errorbar",label="mc smeared 2% mismatch")
# hep.histplot(scalemc_sm.ravel(), ptsZ, yerr=scalemcErr_sm.ravel(), histtype="errorbar",label="mc smeared 5% mismatch")
hep.histplot(scaledata.ravel(), ptsZ, yerr=scaledataErr.ravel(), histtype="errorbar",label="data",ax=ax1)
# hep.histplot(scalemc.ravel(), ptsZ, yerr=scalemcErr.ravel(), histtype="errorbar",label="mc")
ax1.legend(loc='upper left', frameon=False)
hep.histplot(scalemc/scaledata.ravel(), ptsZ, yerr=np.hypot(scaledataErr[...]/scaledata,scalemcErr[...]/scalemc), histtype="errorbar", ax=ax2)
plt.tight_layout()
plt.savefig('scale_1dclosure.png')

# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# plt.hist(scale.ravel(), bins=100, range=[-0.003, 0.003])
# plt.tight_layout()
# plt.savefig('scale_1dclosure.png')

# fig, (ax1) = plt.subplots(figsize=(24,24))
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# # hep.hist2dplot(round(np.array(scaleZ),4),etas,ptsZ, labels =True, cmap = 'jet')
# c = plt.pcolor(np.array(scale), edgecolors='k', cmap='jet',vmin=-0.0017, vmax=0.002)

# def show_values(pc, fmt="%.4f\n +/- %.4f", **kw):
#     pc.update_scalarmappable()
#     ax = pc.axes
#     for p, color, value, err in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array(),scaleErr.ravel()):
#         x, y = p.vertices[:-2, :].mean(0)
#         if np.all(color[:3] > 0.5):
#             color = (0.0, 0.0, 0.0)
#         else:
#             color = (1.0, 1.0, 1.0)
#         ax.text(x, y, fmt % (round(value,4), round(err,4)) , ha="center", va="center",  fontsize=12, **kw)

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
# plt.savefig('scale_2dclosure_afterZ.png')