#include "interface/genLeptonSelector.hpp"
#include <algorithm>
#include <utility>

 float DeltaR(float eta1,float phi1,float eta2,float phi2)
    {
      float deltaPhi = TMath::Abs(phi1-phi2);
      float deltaEta = eta1-eta2;
      if(deltaPhi > TMath::Pi()) deltaPhi = TMath::TwoPi() - deltaPhi;
      return TMath::Sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi);
    };

 ROOT::VecOps::RVec<int> genidxs(const ROOT::VecOps::RVec<float> &Muon_pt, const ROOT::VecOps::RVec<float> &Muon_eta, const ROOT::VecOps::RVec<float> &Muon_phi, const ROOT::VecOps::RVec<int> &Muon_charge, const ROOT::VecOps::RVec<float> &GenPart_pt, const ROOT::VecOps::RVec<float> &GenPart_eta, const ROOT::VecOps::RVec<float> &GenPart_phi, const ROOT::VecOps::RVec<int> &GenPart_pdgId, const ROOT::VecOps::RVec<int> &GenPart_status, const ROOT::VecOps::RVec<int> &GenPart_statusFlags) {
    ROOT::VecOps::RVec<int> res(Muon_pt.size(), -1);
    for (unsigned int i = 0; i < Muon_pt.size(); ++i) {
      
        if (Muon_charge[i] == 99) continue;
        float dr2min = 0.1;
        for (unsigned int j = 0; j < GenPart_pt.size(); ++j) {
            if (GenPart_pdgId[j] != (-Muon_charge[i]*13)) continue;
            if (GenPart_status[j] != 1) continue;
            if ( !(GenPart_statusFlags[j] & 1)) continue;
            
            const float dr2 =DeltaR(Muon_eta[i], Muon_phi[i], GenPart_eta[j], GenPart_phi[j]);
            if (dr2 < dr2min) {
                dr2min = dr2;
                res[i] = j;
            }
        }
    }
    return res;


}

RNode genLeptonSelector::run(RNode d)
{
std::cout<<"genLeptonSelector"<<std::endl;
auto d1 = d.Define("Muon_GenPartIdxCustom", genidxs, {"Muon_cvhPt", "Muon_cvhEta", "Muon_cvhPhi", "Muon_cvhCharge", "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_pdgId", "GenPart_status", "GenPart_statusFlags"})
          .Filter("Muon_GenPartIdxCustom.size()<=2")
          .Define("GoodMuons", "Muon_looseId && Muon_GenPartIdxCustom >= 0")
          .Define("GoodMuons_GenPartIdxCustom", "Muon_GenPartIdxCustom[GoodMuons]")
          .Define("GoodMuons_genEta", "ROOT::VecOps::Take(GenPart_eta, GoodMuons_GenPartIdxCustom)")
          .Define("GoodMuons_genCharge", "Muon_cvhCharge[GoodMuons]")
          .Define("GoodMuons_kgen", "1./ROOT::VecOps::Take(GenPart_pt, GoodMuons_GenPartIdxCustom)");
          // .Define("res", "(GoodMuons_kgen-(1./Muon_cvhPt[GoodMuons]))/GoodMuons_kgen")
          // .Define("GoodMuons_genPtSq","(1./GoodMuons_kgen)*(1./GoodMuons_kgen)");
  return d1;
}
