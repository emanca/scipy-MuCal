#ifndef ZSELECTION_H
#define ZSELECTION_H

#include "module.hpp"

class zSelection : public Module {

    private:

    bool _isData;
    public:
    zSelection(bool isData=false) {
        _isData = isData;
    };
    ~zSelection() {};
    RNode run(RNode) override;

};

#endif
