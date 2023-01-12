#ifndef REWEIGHT_H
#define REWEIGHT_H

#include <iostream>
#include "module.hpp"

class reweight : public Module
{

private:
    TFile *_frew;
    TH2D *hrew;

public:
    reweight(TFile *frew)
    {
        _frew = frew;
        auto hd = (TH2D *)_frew->Get("hpt1pt2d");
        auto hmc = (TH2D *)_frew->Get("hpt1pt2mc");
        hd->Divide(hmc);
        hrew = hd;
    };

    ~reweight(){};
    RNode run(RNode) override;
};

#endif
