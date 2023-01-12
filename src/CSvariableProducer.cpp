#include "interface/CSvariableProducer.hpp"
// #include "functions.hpp"
#include "TLorentzVector.h"
#include "TMath.h"
#include<utility>
RNode CSvariableProducer::run(RNode d) {
  //Logic taken from https://github.com/WMass/nanoAOD-tools/blob/master/python/postprocessing/wmass/CSVariables.py

  auto getCSangles = [](float pt1, float eta1, float phi1, float pt2, float eta2, float phi2)
  {
    std::pair<float, float> cs(-99., -99.);
    TLorentzVector plus, minus;
    plus.SetPtEtaPhiM(pt1, eta1, phi1, 0.105658);
    minus.SetPtEtaPhiM(pt2, eta2, phi2, 0.105658);

    TLorentzVector dilepton(plus + minus);

    // int sign = dilepton.Z() ? std::abs(dilepton.Z())/dilepton.Z() : 0;
    double sign = fabs(dilepton.Z()) / dilepton.Z();

    const float ProtonMass = 0.938272;
    const float BeamEnergy = 6500.000;

    TLorentzVector p1, p2;

    p1.SetPxPyPzE(0, 0,    sign*BeamEnergy, TMath::Sqrt(BeamEnergy*BeamEnergy+ProtonMass*ProtonMass)); 
    p2.SetPxPyPzE(0, 0, -1.*sign*BeamEnergy, TMath::Sqrt(BeamEnergy*BeamEnergy+ProtonMass*ProtonMass));
		  
    p1.Boost(-dilepton.BoostVector());
    p2.Boost(-dilepton.BoostVector());

    auto CSAxis = (p1.Vect().Unit()-p2.Vect().Unit()).Unit(); //quantise along axis that bisects the boosted beams
        
    auto yAxis = (p1.Vect().Unit()).Cross((-p2.Vect().Unit())); //other axes
    yAxis = yAxis.Unit();
    auto xAxis = yAxis.Cross(CSAxis);
    xAxis = xAxis.Unit();
   
    auto boostedLep = minus;
    boostedLep.Boost(-dilepton.BoostVector());
    
    float csphi = TMath::ATan2((boostedLep.Vect()*yAxis),(boostedLep.Vect()*xAxis));
    csphi = csphi<0. ? csphi + 2.*TMath::Pi() : csphi;


    cs.first = TMath::Cos(boostedLep.Angle(CSAxis));
    cs.second = csphi;
    
    return cs;
  };

  auto getCSdeltaphi = [](float pt1, float eta1, float phi1, float pt2, float eta2, float phi2)
  {
    TLorentzVector plus, minus;
    plus.SetPtEtaPhiM(pt1, eta1, phi1, 0.105658);
    minus.SetPtEtaPhiM(pt2, eta2, phi2, 0.105658);

    TLorentzVector dilepton(plus + minus);

    // int sign = dilepton.Z() ? std::abs(dilepton.Z())/dilepton.Z() : 0;
    double sign = fabs(dilepton.Z()) / dilepton.Z();

    const float ProtonMass = 0.938272;
    const float BeamEnergy = 6500.000;

    TLorentzVector p1, p2;

    p1.SetPxPyPzE(0, 0,    sign*BeamEnergy, TMath::Sqrt(BeamEnergy*BeamEnergy+ProtonMass*ProtonMass)); 
    p2.SetPxPyPzE(0, 0, -1.*sign*BeamEnergy, TMath::Sqrt(BeamEnergy*BeamEnergy+ProtonMass*ProtonMass));
		  
    p1.Boost(-dilepton.BoostVector());
    p2.Boost(-dilepton.BoostVector());

    auto CSAxis = (p1.Vect().Unit()-p2.Vect().Unit()).Unit(); //quantise along axis that bisects the boosted beams
        
    auto yAxis = (p1.Vect().Unit()).Cross((-p2.Vect().Unit())); //other axes
    yAxis = yAxis.Unit();
    auto xAxis = yAxis.Cross(CSAxis);
    xAxis = xAxis.Unit();
   
    auto boostedLep1 = minus;
    boostedLep1.Boost(-dilepton.BoostVector());

    auto boostedLep2 = plus;
    boostedLep2.Boost(-dilepton.BoostVector());
    
    double csphi1 = TMath::ATan2((boostedLep1.Vect()*yAxis),(boostedLep1.Vect()*xAxis));
    double csphi2 = TMath::ATan2((boostedLep2.Vect()*yAxis),(boostedLep2.Vect()*xAxis));
    
    double dphi = csphi1 - csphi2;
    if (dphi > M_PI) {
        dphi -= 2.0*M_PI;
    }
    else if (dphi <= - M_PI) {
      dphi += 2.0*M_PI;
    }
    return dphi;
  };

  auto d1 = d.Define("CSAngles", getCSangles, {"Muplus_pt","Muplus_eta","Muplus_phi", "Muminus_pt", "Muminus_eta", "Muminus_phi"})
             .Define("CStheta", [](const std::pair<float, float>& cs){ return cs.first;}, {"CSAngles"})
             .Define("CSphi", [](const std::pair<float, float>& cs){ return cs.second;}, {"CSAngles"})
             .Define("CSdeltaphi",getCSdeltaphi, {"Muplus_pt","Muplus_eta","Muplus_phi", "Muminus_pt", "Muminus_eta", "Muminus_phi"});


  return d1;
}
