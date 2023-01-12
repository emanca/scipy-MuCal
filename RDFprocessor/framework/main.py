import ROOT
from math import pi

qtBins = [0., 4., 8., 12., 16., 20., 24., 28., 32., 40., 60., 100., 200.]
yBins = [0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0, 6.0]
ptBins = [25.+i for i in range(31)]
#overflow-Bin
ptBins.append(200.)
etaBins = [-2.4+i*0.1 for i in range(49)]
cosThetaBins = [round(-1. + 2.*i/100,2) for i in range(101)]
phiBins = [round(0. + 2*pi*i/100,2) for i in range(101)]

bins = [yBins,qtBins,ptBins,etaBins,cosThetaBins,phiBins]

binningCode = 'auto bins = std::make_tuple('
for bin in bins: 
    binningCode+='std::make_tuple({}),'.format(', '.join(str(x) for x in bin))
binningCode = ','.join(binningCode.split(',')[:-1])
binningCode+=')'
#print(binningCode)
ROOT.gInterpreter.ProcessLine(binningCode)

histoCode = '''
#include <boost/histogram.hpp>
#include <memory>
#include <tuple>
#include <utility>
#include <iostream>
#include <chrono>

std::vector<boost::histogram::axis::variable<>> v;

template <typename BinsForOneAxis>
void CreateAxis(BinsForOneAxis &b)
{{
    auto axis_lambda = [&](auto &&...args) {{ return boost::histogram::axis::variable<>{{args...}}; }};
    auto axis = std::apply(axis_lambda, b);
    v.emplace_back(axis);
}}

template <typename... Args, std::size_t... N>
void bins_helper(const std::tuple<Args...> &b, std::index_sequence<N...>)
{{
    (CreateAxis(std::get<N>(b)), ...);
}}

template <typename Args, typename Idx = std::make_index_sequence<std::tuple_size<std::decay_t<Args>>::value>>
void construct_with_bins(const Args &b)
{{
    bins_helper(b, Idx{{}});
}}

template<typename Bins>
void createHisto(Bins bins){{
    // set up the binning
    auto b1 = std::get<0>(bins);
    auto d1 = std::tuple_size<decltype(b1)>::value - 1;
    auto b2 = std::get<1>(bins);
    auto d2 = std::tuple_size<decltype(b2)>::value - 1;
    auto b3 = std::get<2>(bins);
    auto d3 = std::tuple_size<decltype(b3)>::value - 1;
    auto b4 = std::get<3>(bins);
    auto d4 = std::tuple_size<decltype(b4)>::value - 1;
    auto b5 = std::get<4>(bins);
    auto d5 = std::tuple_size<decltype(b5)>::value - 1;
    auto b6 = std::get<5>(bins);
    auto d6 = std::tuple_size<decltype(b6)>::value - 1;
    int nbins = d1 * d2 * d3 * d4 * d5 * d6;
    std::cout << d1 << " " << d2 << " " << d3 << " " << d4 << " " << d5 << " " << d6 << std::endl;
    std::cout << "number of bins is " << nbins << std::endl;
    construct_with_bins(bins);
    auto start = std::chrono::steady_clock::now();
    auto htmp = boost::histogram::make_weighted_histogram(v);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << std::endl;
    std::cout << "rank is " << htmp.rank() << std::endl;
}}

void callFunction({binsType} bins){{
    createHisto<{binsType}>(bins);
}}

'''
templ = type(ROOT.bins).__cpp_name__
with open("helperbooker.cpp", "w") as f:
    f.write(histoCode.format(binsType=templ))                                                                                                   
ROOT.gSystem.CompileMacro("helperbooker.cpp", "gfkO")

ROOT.callFunction(ROOT.bins)

