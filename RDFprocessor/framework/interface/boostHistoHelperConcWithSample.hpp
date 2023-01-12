#ifndef BOOSTHISTOHELPERCONCWITHSAMPLE_H
#define BOOSTHISTOHELPERCONCWITHSAMPLE_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <boost/histogram.hpp>
#include <type_traits>
#include <boost/format.hpp> // only needed for printing
#include <boost/functional/hash.hpp>
#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for thread_safe_withvariance_sample<>
#include "thread_safe_withvariance_sample.hpp"
#include <memory>
#include <tuple>
#include <utility>
#include <chrono>

template <std::size_t Ncols, std::size_t Nweights, std::size_t Dsample, typename Bins>
class boostHistoHelperConcWithSample : public ROOT::Detail::RDF::RActionImpl<boostHistoHelperConcWithSample<Ncols, Nweights, Dsample, Bins>>
{

public:
   // shortcut for complex boost histogram type
   using boost_histogram = boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::thread_safe_withvariance_sample<double, Dsample>, std::allocator<boost::histogram::accumulators::thread_safe_withvariance_sample<double, Dsample>>>>>;
   // This type is a requirement for every RDF helper.
   using Result_t = std::map<std::string, boost_histogram>;

private:
   std::shared_ptr<std::map<std::string, boost_histogram>> fHistos;
   std::vector<boost_histogram *> _histoPtrs;
   std::vector<std::vector<std::string>> _variationRules;   // to keep track of the variations --> ordered as columns
   std::string _name;                                       // name of histogram
   std::vector<boost::histogram::axis::variable<>> _v;      // custom axes of histogram
   std::vector<std::vector<float>> _columns;                // one value per column per processing slot
   std::vector<std::vector<float>> _weights;                // one value per weight per processing slot
   std::vector<std::vector<double>> _samples;               // one sample per processing slot: must be double as per boost request
   std::vector<std::vector<ROOT::RVec<float>>> _variations; // one RVec per column variation per processing slot
   std::vector<std::size_t> _colsWithVariationsIdx;         // indices of the columns with a variation

public:
   boostHistoHelperConcWithSample(std::string name, std::vector<std::vector<std::string>> variationRules, Bins bins, unsigned int nSlots) : _columns{nSlots}, _weights{nSlots}, _variations{nSlots}, _samples{nSlots}, _name{name}, _variationRules{variationRules}
   {
      for (auto &c : _columns)
         c.resize(Ncols);
      for (auto &w : _weights)
         w.resize(Nweights);
      for (auto &v : _variations)
         v.resize(Ncols); // i.e. as big as _variationRules.size(): encodes possible column value variations
      // declare the binning at compile time for best performance
      construct_with_bins(bins);
      std::make_index_sequence<std::tuple_size<decltype(bins)>::value> idx;

      // save index number (in the full list of columns) of the next column _with a variation_
      int colIdx = 0;
      for (auto &groupOfVars : _variationRules)
      {
         if (groupOfVars[0] == "")
         {
            ++colIdx;
            continue;
         }
         _colsWithVariationsIdx.emplace_back(colIdx);
         ++colIdx;
      }

      // set up one single histogram to be filled by all threads concurrently
      fHistos = std::make_shared<std::map<std::string, boost_histogram>>();
      std::map<std::string, boost_histogram> &hmap = *fHistos;

      // first make nominal histogram
      std::cout << "creating nominal " << std::endl;
      auto htmp = boost::histogram::make_histogram_with(boost::histogram::dense_storage<boost::histogram::accumulators::thread_safe_withvariance_sample<double, Dsample>>(), _v);
      auto it = hmap.insert(std::make_pair(_name, htmp));
      _histoPtrs.emplace_back(&(it.first->second)); // address of the thing just inserted
   }                                                // end constructor

   boostHistoHelperConcWithSample(boostHistoHelperConcWithSample &&) = default;
   boostHistoHelperConcWithSample(const boostHistoHelperConcWithSample &) = delete;
   std::shared_ptr<std::map<std::string, boost_histogram>> GetResultPtr() const { return fHistos; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   template <typename BinsForOneAxis>
   void CreateAxis(BinsForOneAxis &b)
   {
      auto axis_lambda = [&](auto &&...args)
      { return boost::histogram::axis::variable<>{args...}; };
      auto axis = std::apply(axis_lambda, b);
      _v.emplace_back(axis);
   }
   template <typename... Args, std::size_t... N>
   void bins_helper(const std::tuple<Args...> &b, std::index_sequence<N...>)
   {
      (CreateAxis(std::get<N>(b)), ...);
   }

   template <typename Args, typename Idx = std::make_index_sequence<std::tuple_size<std::decay_t<Args>>::value>>
   void construct_with_bins(const Args &b)
   {
      bins_helper(b, Idx{});
   }

   void construct_with_bins(const std::vector<std::vector<float>> &bins)
   {
      for (auto &b : bins)
         _v.emplace_back(b);
   }

   void FillValues(float val, std::size_t n, unsigned int nSlot)
   {
      if (n < Ncols)
      {
         _columns[nSlot][n] = val;
      }
      else
      {
         _weights[nSlot][n - Ncols] = val;
      }
   }

   void FillValues(const ROOT::RVec<float> &val, std::size_t n, unsigned int nSlot)
   {
      std::vector<double> v(val.begin(), val.end());
      _samples[nSlot] = v;
   }

   template <std::size_t... Is>
   void FillBoostHisto(boost_histogram &h, float weight, const std::vector<float> &columns, std::vector<double> sample, std::index_sequence<Is...>)
   {
      h(boost::histogram::weight(weight), boost::histogram::sample(sample), columns[Is]...);
   }

   template <typename... Ts>
   void Exec(unsigned int slot, const Ts &...cols)
   {
      std::vector<boost_histogram *> &histos = _histoPtrs;

      //extract columns, weights and variations from cols
      std::size_t i = 0;
      (FillValues(cols, i++, slot), ...);

      auto &columns = _columns[slot];
      auto &weights = _weights[slot];
      auto &samples = _samples[slot];

      float weight = std::accumulate(std::begin(weights), std::end(weights), 1.f, std::multiplies<float>());
      auto *h = histos[0];
      FillBoostHisto(*h, weight, columns, samples, std::make_index_sequence<Ncols>{});
   }

   void Finalize()
   {
      // auto &map = *fHistos;
      // for (auto &y : map)
      // {
      //    for (auto &&x : boost::histogram::indexed(map.at(y.first)))
      //    {
      //       std::vector<std::array<double, Dsample>> vals;
      //       std::vector<std::array<double, Dsample>> sumw2;
      //       const auto &n = x->value(); //this is a std::array<atomic<double>,D>
      //       const auto &w2 = x->variance();
      //       std::array<double, Dsample> n_doubles;
      //       std::array<double, Dsample> w2_doubles;
      //       // n_doubles.resize(Dsample);
      //       // w2_doubles.resize(Dsample);
      //       for (unsigned int j = 0; j < Dsample; j++)
      //       {
      //          n_doubles[j] = n[j].load();
      //          w2_doubles[j] = w2[j].load();
      //          // std::cout << n_doubles[j] << std::endl;
      //       }
      //       vals.emplace_back(n_doubles);
      //       sumw2.emplace_back(w2_doubles);
      //    }
      // }
   }
   std::string GetActionName()
   {
      return "boostHistoHelperConcWithSample";
   }
};

#endif
