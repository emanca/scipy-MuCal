#ifndef SF_UL_H
#define SF_UL_H

#include "module.hpp"
#include "TString.h"
#include <iostream>
class SF_ul : public Module
{

private:
  TFile *_SF;
  TH2D *_tracking;
  TH2D *_reco;
  TH2D *_idip;
  TH2D *_trigger_plus;
  TH2D *_trigger_minus;
  TH2D *_iso;
  TH2D *_iso_notrig;
  TH2D *_antiiso;
  TH2D *_effisomc;
  TH2D *_effisodata;

  TH2D *_tracking_alt;
  TH2D *_reco_alt;
  TH2D *_idip_alt;
  TH2D *_trigger_plus_alt;
  TH2D *_trigger_minus_alt;
  TH2D *_iso_alt;
  TH2D *_iso_notrig_alt;
  TH2D *_antiiso_alt;

  bool _isZ;
  // this is only relevant for Z studies
  int _prefCharge;

public:
  SF_ul(TFile *SF, bool isZ = false, std::string era = "preVFP", int prefCharge = 1)
  {
    _SF = SF;
    _isZ = isZ;
    TString tag = "BtoH"; // for all 2016
    if (era == "preVFP")
      tag = "BtoF";
    else if (era == "postVFP")
      tag = "GtoH";
    TString version = "nominal";
    
    std::cout << "SF tag is " << tag << std::endl;
    _reco = (TH2D *)_SF->Get("SF2D_"+ version + "_reco_" + tag + "_both");
    _tracking = (TH2D *)_SF->Get("SF2D_" + version + "_tracking_" + tag + "_both");
    _idip = (TH2D *)_SF->Get("SF2D_" + version + "_idip_" + tag + "_both");
    _trigger_plus = (TH2D *)_SF->Get("SF2D_" + version + "_trigger_" + tag + "_plus");
    _trigger_minus = (TH2D *)_SF->Get("SF2D_" + version + "_trigger_" + tag + "_minus");
    _iso = (TH2D *)_SF->Get("SF2D_" + version + "_iso_" + tag + "_both");
    _iso_notrig = (TH2D *)_SF->Get("SF2D_" + version + "_isonotrig_" + tag + "_both");
    _antiiso = (TH2D *)_SF->Get("SF2D_" + version + "_antiiso_" + tag + "_both");
    _effisomc = (TH2D *)_SF->Get("effMC_iso_" + tag + "_both");
    _effisodata = (TH2D *)_SF->Get("effData_iso_" + tag + "_both");

    version = "dataAltSig";
    _reco_alt = (TH2D *)_SF->Get("SF2D_" + version + "_reco_" + tag + "_both");
    _tracking_alt = (TH2D *)_SF->Get("SF2D_" + version + "_tracking_" + tag + "_both");
    _idip_alt = (TH2D *)_SF->Get("SF2D_" + version + "_idip_" + tag + "_both");
    _trigger_plus_alt = (TH2D *)_SF->Get("SF2D_" + version + "_trigger_" + tag + "_plus");
    _trigger_minus_alt = (TH2D *)_SF->Get("SF2D_" + version + "_trigger_" + tag + "_minus");
    _iso_alt = (TH2D *)_SF->Get("SF2D_" + version + "_iso_" + tag + "_both");
    _iso_notrig_alt = (TH2D *)_SF->Get("SF2D_" + version + "_isonotrig_" + tag + "_both");
    _antiiso_alt = (TH2D *)_SF->Get("SF2D_" + version + "_antiiso_" + tag + "_both");
    
    _prefCharge = prefCharge;
  };
  ~SF_ul(){};

  RNode run(RNode) override;
};

#endif
