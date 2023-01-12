#include "applyCalibration.hpp"

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

  auto smear = [myRndGens](unsigned int slot, float pt) mutable
  {
    return pt; //+ myRndGens[slot].Gaus(0., 0.01 * pt);
  };

  auto correct = [this](float pt, float eta, float phi, int charge) -> float
  {
    float curvature = 1.0 / getCorrectedPtMag(pt, eta, phi);
    int bin = hA->FindBin(eta);
    float a = hA->GetBinContent(bin);
    float e = he->GetBinContent(bin);
    float m = hM->GetBinContent(bin);

    float magnetic = 1. + a;
    float material = -e * curvature;
    float alignment = charge * m;
    curvature = (magnetic + material) * curvature + alignment;
    return 1.0 / curvature;
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
                  .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(pt1corr,Muplus_eta,Muplus_phi, 0.105658)")
                  .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(pt2corr,Muminus_eta,Muminus_phi, 0.105658)")
                  .Define("corrMass", "float((v1corr+v2corr).M())");
    return d1;
  }
}
