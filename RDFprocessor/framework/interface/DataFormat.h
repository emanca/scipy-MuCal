#ifndef DATAFORMAT_H
#define DATAFORMAT_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <boost/histogram.hpp>
#include "boostHistoHelperWithSample.hpp"
#include "boostHistoHelperConcWithSample.hpp"
#include "boostHistoHelper.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <iostream>

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

template <typename boost_histogram, std::size_t Ncols, std::size_t Nweights, std::size_t Dsample, std::size_t totsize, typename Bins, typename... Ts>
struct Histogram
{
    ROOT::RDF::RResultPtr<std::map<std::string, boost_histogram>> operator()(RNode d, std::string name, Bins bins, const std::vector<std::string> &columns, std::vector<std::vector<std::string>> variationRules)
    {
        if constexpr (totsize > 1.e6) // 100M bins threshold
        {
            if constexpr (Dsample == 1)
            {
                std::cout << "call helper..." << std::endl;
                boostHistoHelper<Ncols, Nweights, Bins> helper(name, variationRules, bins, d.GetNSlots());
                std::cout << "call book..." << std::endl;
                auto h = d.Book<Ts...>(std::move(helper), columns);
                return h;
            }
            else
            {
                std::cout << "call concurrent helper with sample..." << std::endl;
                boostHistoHelperConcWithSample<Ncols, Nweights, Dsample, Bins> helper(name, variationRules, bins, d.GetNSlots());
                std::cout << "call book..." << std::endl;
                auto h = d.Book<Ts...>(std::move(helper), columns);
                return h;
            }
        }
        else
        {
            if constexpr (Dsample == 1)
            {
                std::cout << "call helper..." << std::endl;
                boostHistoHelper<Ncols, Nweights, Bins> helper(name, variationRules, bins, d.GetNSlots());
                std::cout << "call book..." << std::endl;
                auto h = d.Book<Ts...>(std::move(helper), columns);
                return h;
            }
            else
            {
                std::cout << "call helper with sample..." << std::endl;
                boostHistoHelperWithSample<Ncols, Nweights, Dsample, Bins> helper(name, variationRules, bins, d.GetNSlots());
                std::cout << "call book..." << std::endl;
                auto h = d.Book<Ts...>(std::move(helper), columns);
                return h;
            }
        }
    }
};

#endif
