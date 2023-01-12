#ifndef APPLYCALIBRATION_H
#define APPLYCALIBRATION_H

#include <iostream>
#include "module.hpp"
#include "TRandom3.h"

class applyCalibration : public Module
{

private:
    TFile *_calib;
    TFile *_bFieldMap;
    TH2F *hbFieldMap;
    TH1D *hA;
    TH1D *he;
    TH1D *hM;

    bool _isData;
    bool _fullCalib;
    bool _layerCorr;

public:
    applyCalibration(TFile *bFieldMap, TFile *calib, bool isData = false, bool fullCalib = false, bool layerCorr = false)
    {
        _bFieldMap = bFieldMap;
        _calib = calib;
        hbFieldMap = (TH2F *)_bFieldMap->Get("bfieldMap");
        hA = (TH1D *)_calib->Get("A");
        he = (TH1D *)_calib->Get("e");
        hM = (TH1D *)_calib->Get("M");
        _isData = isData;
        _fullCalib = fullCalib;
        _layerCorr = layerCorr;
    };

    ~applyCalibration(){};
    RNode run(RNode) override;
    float getCorrectedPtMag(float, float, float);
};

#endif
