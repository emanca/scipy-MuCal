#ifndef BOOSTHISTOHELPER_H
#define BOOSTHISTOHELPER_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <boost/histogram.hpp>
#include <boost/format.hpp> // only needed for printing
#include <boost/functional/hash.hpp>
#include <memory>
#include <tuple>
#include <utility>
#include <chrono>

template <std::size_t Ncols, std::size_t Nweights, typename Bins>
class boostHistoHelper : public ROOT::Detail::RDF::RActionImpl<boostHistoHelper<Ncols, Nweights, Bins>>
{

public:
   using boost_histogram = boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum<>, std::allocator<boost::histogram::accumulators::weighted_sum<>>>>>;
   /// This type is a requirement for every helper.
   using Result_t = std::map<std::string, boost_histogram>;

private:
   std::vector<std::shared_ptr<std::map<std::string, boost_histogram>>> fHistos; // one per data processing slot
   std::vector<std::vector<boost_histogram *>> _histoPtrs;                       // one per data processing slot, pointint to the contents of fHistos (used for quicker access)
   std::vector<std::vector<std::string>> _variationRules;                        //to keep track of the variations --> ordered as columns
   std::string _name;
   std::vector<boost::histogram::axis::variable<>> _v;
   std::vector<std::vector<float>> _columns;                // one value per column per processing slot
   std::vector<std::vector<float>> _weights;                // one value per weight per processing slot
   std::vector<std::vector<ROOT::RVec<float>>> _variations; // one RVec per variation per processing slot
   std::vector<std::vector<float>> _columns_var;
   std::vector<std::vector<float>> _weights_var;
   std::vector<std::size_t> _colsWithVariationsIdx;

public:
   /// This constructor takes all the parameters necessary to build the THnTs. In addition, it requires the names of
   /// the columns which will be used.
   boostHistoHelper(std::string name, std::vector<std::vector<std::string>> variationRules, Bins bins, unsigned int nSlots) : _histoPtrs{nSlots}, _columns{nSlots}, _weights{nSlots}, _variations{nSlots}, _columns_var{nSlots}, _weights_var{nSlots}, _name{name}, _variationRules{variationRules}
   {
      for (auto &c : _columns)
         c.resize(Ncols);
      for (auto &w : _weights)
         w.resize(Nweights);
      for (auto &v : _variations)
         v.resize(Ncols + Nweights); // i.e. as big as _variationRules.size()
      for (auto &c : _columns_var)
         c.resize(Ncols);
      for (auto &w : _weights_var)
         w.resize(Ncols + Nweights);

      construct_with_bins(bins);

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

      for (auto slot : ROOT::TSeqU(nSlots))
      {
         fHistos.emplace_back(std::make_shared<std::map<std::string, boost_histogram>>());
         (void)slot;

         std::map<std::string, boost_histogram> &hmap = *fHistos[slot];

         std::string slotnum = "";
         slotnum = slot > 0 ? std::to_string(slot) : "";
         // first make nominal histogram
         // std::cout << "creating nominal " << std::endl;
         auto start = std::chrono::steady_clock::now();
         auto htmp = boost::histogram::make_weighted_histogram(_v);
         auto end = std::chrono::steady_clock::now();
         std::chrono::duration<double> elapsed_seconds = end - start;
         // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
         // std::cout << "rank is " << htmp.rank() << std::endl;
         auto it = hmap.insert(std::make_pair(_name, htmp));
         // std::cout << "inserted nominal " << std::endl;
         _histoPtrs[slot].emplace_back(&(it.first->second)); // address of the thing just inserted
         // std::cout << "nominal in vector" << std::endl;
         //then check variations
         for (auto &groupOfVars : _variationRules)
         {
            if (groupOfVars[0] == "")
               continue;
            for (auto &var : groupOfVars)
            {
               auto htmp = boost::histogram::make_weighted_histogram(_v);
               std::string histoname = _name + "_" + var;
               // std::cout << "histoname " << histoname << std::endl;
               auto it = hmap.insert(std::make_pair(histoname, htmp));
               _histoPtrs[slot].emplace_back(&(it.first->second)); // address of the thing just inserted
            }
         }
      }
   }

   boostHistoHelper(boostHistoHelper &&) = default;
   boostHistoHelper(const boostHistoHelper &) = delete;
   std::shared_ptr<std::map<std::string, boost_histogram>> GetResultPtr() const { return fHistos[0]; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   template <typename BinsForOneAxis>
   void CreateAxis(BinsForOneAxis &b)
   {
      auto axis_lambda = [&](auto &&...args) {return boost::histogram::axis::variable<>{args...}; };
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
      // convert "idx in array of cols with variations" into "idx in array of all cols"
      const auto nColOutOfAllColumns = _colsWithVariationsIdx[n - Ncols - Nweights];
      _variations[nSlot][nColOutOfAllColumns] = val;
   }

   template <std::size_t... Is>
   void FillBoostHisto(boost_histogram &h, float weight, const std::vector<float> &columns, std::index_sequence<Is...>)
   {
      h(boost::histogram::weight(weight), columns[Is]...);
   }

   template <typename... Ts>
   void Exec(unsigned int slot, const Ts &...cols)
   {
      std::vector<boost_histogram *> &histos = _histoPtrs[slot];

      //extract columns, weights and variations from cols
      std::size_t i = 0;
      (FillValues(cols, i++, slot), ...);

      auto &columns = _columns[slot];
      auto &weights = _weights[slot];

      float weight = std::accumulate(std::begin(weights), std::end(weights), 1.f, std::multiplies<float>());
      auto *h = histos[0];
      // std::cout << "filling nominal " << std::endl;
      FillBoostHisto(*h, weight, columns, std::make_index_sequence<Ncols>{});

      auto &columns_var = _columns_var[slot];
      auto &weights_var = _weights_var[slot];
      auto &variationVecs = _variations[slot];

      // now fill variations
      int nHistogram = 1; // skip the nominal
      for (auto i : _colsWithVariationsIdx)
      {
         // this index will tell which column to vary
         for (unsigned int j = 0; j < _variationRules[i].size(); j++)
         {
            // first copy the nominal vector every time
            columns_var = columns;
            weights_var = weights;
            // substitute the relevant column with its variation and fill
            if (i < (Ncols + Nweights))
            {
               if (i < Ncols)
                  columns_var[i] = variationVecs[i][j];
               else
                  weights_var[i - Ncols] = variationVecs[i][j];
            }
            else
               throw std::invalid_argument("you're trying to vary a variation...");
            float weight = std::accumulate(std::begin(weights_var), std::end(weights_var), 1.f, std::multiplies<float>());
            auto *h = histos[nHistogram];
            FillBoostHisto(*h, weight, columns_var, std::make_index_sequence<Ncols>{});
            ++nHistogram;
         }
      }
   }

   void Finalize()
   {
      // std::cout << "in Finalize" << std::endl;
      auto &res = *fHistos[0];
      for (auto slot : ROOT::TSeqU(1, fHistos.size()))
      {
         auto &map = *fHistos[slot];
         for (auto &x : res)
         {
            x.second += map.at(x.first);
            // std::cout << x.second.rank() << std::endl;
         }
      }
   }
   std::string GetActionName()
   {
      return "boostHistoHelper";
   }
};

#endif
