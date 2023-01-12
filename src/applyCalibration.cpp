#include "interface/applyCalibration.hpp"

float applyCalibration::getCorrectedPtMag(float pt, float eta, float phi)
{

  float magneticFactor = 1.0;
  if (_isData)
    magneticFactor = magneticFactor * hbFieldMap->GetBinContent(hbFieldMap->GetBin(hbFieldMap->GetXaxis()->FindBin(phi), hbFieldMap->GetYaxis()->FindBin(eta)));
  float curvature = (magneticFactor) / pt;

  return 1.0 / curvature;
}

RNode applyCalibration::run(RNode d)
{
  auto NSlots = d.GetNSlots();
  std::vector<TRandom3> myRndGens(NSlots);
  int seed = 1; // not 0 because seed 0 has a special meaning
  for (auto &&gen : myRndGens)
  {
    gen.SetSeed(seed++);
  }

  // auto smear = [this, myRndGens](unsigned int slot, float pt, float eta) mutable -> float
  // {
  //   if (!_isData)
  //   {
  //     int xbin = std::max(1, std::min(hsmear->GetNbinsX(), hsmear->GetXaxis()->FindFixBin(eta)));
  //     int ybin = std::max(1, std::min(hsmear->GetNbinsY(), hsmear->GetYaxis()->FindFixBin(pt)));
  //     float sigma = hsmear->GetBinContent(xbin, ybin);
  //     if (sigma > 0.)
  //       return float(pt + myRndGens[slot].Gaus(0., sigma * pt));
  //     else
  //     {
  //       return pt;
  //     }
  //   }
  //   else
  //   {
  //     int xbin = std::max(1, std::min(hsmear->GetNbinsX(), hsmear->GetXaxis()->FindFixBin(eta)));
  //     int ybin = std::max(1, std::min(hsmear->GetNbinsY(), hsmear->GetYaxis()->FindFixBin(pt)));
  //     float sigma = hsmear->GetBinContent(xbin, ybin);
  //     if (sigma < 0.)
  //     {
  //       sigma = -1. * sigma;
  //       return float(pt + myRndGens[slot].Gaus(0., sigma * pt));
  //     }
  //     else
  //     {
  //       return pt;
  //     }
  //   }
  // };

   auto smear = [this, myRndGens](unsigned int slot, float pt, float eta, float res) mutable -> float
  {
    if (!_isData)
    {
      float smearedk = 1./pt + myRndGens[slot].Gaus(0., 0.5*res);
      return 1./smearedk;
    }
    return pt;
  };

  auto correct = [this](float pt, float eta, float phi, int charge) -> float
  {
    float curvature = 1.0 / getCorrectedPtMag(pt, eta, phi);
    int bin = hA->FindBin(eta);
    float a = hA->GetBinContent(bin);
    float e = he->GetBinContent(bin);
    // float m = hM->GetBinContent(bin);
    float m = 0.;
    float magnetic = 1. + a;
    float material = -e * curvature;
    float alignment = charge * m;
    // if (!_isData) alignment-= charge * mZ;
    curvature = (magnetic + material) * curvature + alignment;
    return 1.0 / curvature;
  };

  auto correctPhi = [this](float eta, float phi) -> float
  {
    int bin = hMphi->FindBin(eta);
    float m_phi = hMphi->GetBinContent(bin);
    float alignment = m_phi;
    float phi_corr = phi + alignment;
    return phi_corr;
  };
  auto correctTheta = [this](float eta) -> float
  {
    int bin = hMtheta->FindBin(eta);
    float m_theta = hMtheta->GetBinContent(bin);
    float alignment = m_theta;
    float theta = 2*TMath::ATan(TMath::Exp(-eta));
    float theta_corr = theta - alignment;
    return theta_corr;
  };

  auto lambda = [this](float pt, float eta, float phi)
  {
    return getCorrectedPtMag(pt, eta, phi);
  };

  std::string corrs = _layerCorr ? "cor" : "";

  if (!(_fullCalib))
  {
    std::cout << "calibmap" << std::endl;
    auto d1 = d.Define("pt1corr", lambda, {Form("Muplus%s_pt", corrs.c_str()), Form("Muplus%s_eta", corrs.c_str()), Form("Muplus%s_phi", corrs.c_str())})
                  .Define("pt2corr", lambda, {Form("Muminus%s_pt", corrs.c_str()), Form("Muminus%s_eta", corrs.c_str()), Form("Muminus%s_phi", corrs.c_str())})
                  .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(pt1corr,Muplus_eta,Muplus_phi, 0.105658)")
                  .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(pt2corr,Muminus_eta,Muminus_phi, 0.105658)")
                  .Define("Jpsi_rap", "float((v1corr+v2corr).Rapidity())")
                  .Define("corrMass", "float((v1corr+v2corr).M())");
    return d1;
  }
  else
  {
    std::cout << "full calib" << std::endl;
    auto d1 = d.Define("charge1", "1")
                  .Define("charge2", "-1")
                  .Define("pt1corr", correct, {Form("Muplus%s_pt", corrs.c_str()), Form("Muplus%s_eta", corrs.c_str()), Form("Muplus%s_phi", corrs.c_str()), "charge1"})
                  .Define("pt2corr", correct, {Form("Muminus%s_pt", corrs.c_str()), Form("Muminus%s_eta", corrs.c_str()), Form("Muminus%s_phi", corrs.c_str()), "charge2"})
                  // .Define("phi1corr",correctPhi,{"Muplus_eta","Muplus_phi"})
                  // .Define("phi2corr",correctPhi,{"Muminus_eta","Muminus_phi"})
                  // .Define("theta1corr",correctTheta,{"Muplus_eta"})
                  // .Define("theta2corr",correctTheta,{"Muminus_eta"})
                  // .Define("eta1corr","-TMath::Log(TMath::Tan(theta1corr/2))")
                  // .Define("eta2corr","-TMath::Log(TMath::Tan(theta2corr/2))")
                  // .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(pt1corr,eta1corr,phi1corr, 0.105658)")
                  // .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(pt2corr,eta2corr,phi2corr, 0.105658)")
                  // .Define("Jpsi_rap", "float((v1corr+v2corr).Rapidity())")
                  // .Define("corrMass", "float((v1corr+v2corr).M())");
                  .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(pt1corr,Muplus_eta,Muplus_phi, 0.105658)")
                  .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(pt2corr,Muminus_eta,Muminus_phi, 0.105658)")
                  .Define("Jpsi_rap", "float((v1corr+v2corr).Rapidity())")
                  .Define("corrMass", "float((v1corr+v2corr).M())");
    if(!_isData){
                  d1=d1.Define("k1corrsm","1./pt1corr+myRndGens[rdfslot_].Gaus(0.,Err1)")
                  .Define("k2corrsm","1./pt2corr+myRndGens[rdfslot_].Gaus(0.,Err2)")
                  .Define("pt1corrsm", "1./k1corrsm")
                  .Define("pt2corrsm", "1./k2corrsm")
                  .Define("v1corrsm", "ROOT::Math::PtEtaPhiMVector(pt1corrsm,Muplus_eta,Muplus_phi, 0.105658)")
                  .Define("v2corrsm", "ROOT::Math::PtEtaPhiMVector(pt2corrsm,Muminus_eta,Muminus_phi, 0.105658)")
                  .Define("corrMasssm", "float((v1corrsm+v2corrsm).M())")
                  .Define("respluscorr", "float((1./pt1corr - 1./Muplusgen_pt)/(1./Muplusgen_pt))")
                  .Define("resminuscorr", "float((1./pt2corr - 1./Muminusgen_pt)/(1./Muminusgen_pt))");
    }
    return d1;
  }
}
