#include "genVProducer.hpp"
#include "TLorentzVector.h"

RNode genVProducer::run(RNode d)
{
        //Logic equivalent to  https://github.com/WMass/nanoAOD-tools/blob/master/python/postprocessing/wmass/Vproducer.py
        //but Line 14 - how is it ensured that the idx 1 is muons?
        //to be safe I am using GenPart_mass
        auto getGenV = [](const RVec<float> &pt, const RVec<float> &eta, const RVec<float> &phi, const RVec<float> &mass, const int idx1, const int idx2)
        {
                if (idx1 < 0 || idx2 < 0)
                {
                        TLorentzVector tmp;
                        tmp.SetPtEtaPhiE(-99., -99., -99., -99.);
                        return tmp;
                }

                TLorentzVector plus, minus;
                plus.SetPtEtaPhiM(pt[idx1], eta[idx1], phi[idx1], mass[idx1]);
                minus.SetPtEtaPhiM(pt[idx2], eta[idx2], phi[idx2], mass[idx2]);
                TLorentzVector dilepton(plus + minus);
                return dilepton;
        };

        auto getComp = [](const RVec<float> &vec, const int idx1) ->float
        {
                if (idx1 < 0)
                        return -99;
                else
                {
                        return vec[idx1];
                }
        };

        auto d1 = d.Define("GenVP4", getGenV, {"GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_postFSRLepIdx1", "GenPart_postFSRLepIdx2"})
                      .Define("Muplusgen_pt", getComp, {"GenPart_pt", "GenPart_postFSRLepIdx1"})
                      .Define("Muminusgen_pt", getComp, {"GenPart_pt", "GenPart_postFSRLepIdx2"})
                      .Define("Muplusgen_eta", "GenPart_eta[GenPart_postFSRLepIdx1]")
                      .Define("Muminusgen_eta", "GenPart_eta[GenPart_postFSRLepIdx2]")
                      .Define("Vpt_postFSR", [this](TLorentzVector p)
                              { return float(p.Pt()); },
                              {"GenVP4"})
                      .Define("Vrap_postFSR", [this](TLorentzVector p)
                              { return float(p.Rapidity()); },
                              {"GenVP4"})
                      .Define("Vrap_postFSR_abs", "TMath::Abs(Vrap_postFSR)")
                      .Define("Vphi_postFSR", [this](TLorentzVector p)
                              { return float(p.Phi()); },
                              {"GenVP4"})
                      .Define("Vmass_postFSR", [this](TLorentzVector p)
                              { return float(p.M()); },
                              {"GenVP4"})
                      .Alias("Jpsigen_mass", "Vmass_postFSR");

        return d1;
}
