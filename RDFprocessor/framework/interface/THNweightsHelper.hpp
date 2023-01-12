#ifndef THNWEIGHTSHELPER_H
#define THNWEIGHTSHELPER_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "THn.h"

class THNweightsHelper : public ROOT::Detail::RDF::RActionImpl<THNweightsHelper>
{

public:
   /// This is a handy, expressive shortcut.
   using THn_t = THnT<float>;
   /// This type is a requirement for every helper.
  using Result_t = std::vector<std::unique_ptr<THn_t>>;
private:
  std::vector<std::shared_ptr<std::vector<std::unique_ptr<THn_t>>>> fHistos; // one per data processing slot
   std::string _name;
   std::array<int, 5> _nbins;
   std::array<double, 5> _xmins;
   std::array<double, 5> _xmax;
   std::vector<std::string> _weightNames;
  

public:
   /// This constructor takes all the parameters necessary to build the THnTs. In addition, it requires the names of
   /// the columns which will be used.
   THNweightsHelper(std::string name, std::string title,
                    std::array<int, 5> nbins, std::array<double, 5> xmins,
                    std::array<double, 5> xmax,
                    std::vector<std::string> weightNames);

   THNweightsHelper(THNweightsHelper &&) = default;
   THNweightsHelper(const THNweightsHelper &) = delete;
  std::shared_ptr<std::vector<std::unique_ptr<THn_t>>> GetResultPtr() const;
   void Initialize();
   void InitTask(TTreeReader *, unsigned int);
   /// This is a method executed at every entry
   void Exec(unsigned int slot, const float &var1, const float &var2, const float &var3, const float &var4, const float &var5, const float &weight, const ROOT::VecOps::RVec<float> &weights);
   void Finalize();
   std::string GetActionName();
};

#endif

