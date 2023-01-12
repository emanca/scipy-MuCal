#ifndef UTILITY_H
#define UTILITY_H
#include </usr/include/eigen3/Eigen/Core>
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/include/eigen3/Eigen/Sparse>
#include <vector>
#include <boost/histogram.hpp>
#include <chrono>
#include "weighted_sum_vec.hpp"
#include "thread_safe_withvariance_sample.hpp"

using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

template <typename boost_histogram>
auto convert(const boost_histogram &h)
{
    for (auto &&x : indexed(h))
    {
        auto testVar = x->value();
        if constexpr (std::is_convertible_v<decltype(testVar), float>)
        {
            std::vector<float> vals;
            std::vector<float> sumw2;
            for (auto &&x : indexed(h))
            {
                const auto n = x->value();
                const auto w2 = x->variance();
                vals.emplace_back(n);
                sumw2.emplace_back(w2);
            }
            return std::vector<std::vector<float>>{vals, sumw2};
        }
        else
        {
            Eigen::initParallel();
            std::vector<decltype(testVar)> vals;
            std::vector<decltype(testVar)> sumw2;
            for (auto &&x : indexed(h))
            {
                const auto n = x->value(); //this is a std::array<double,D>
                const auto w2 = x->variance();
                vals.emplace_back(n);
                sumw2.emplace_back(w2);
            }
            int size = sizeof(vals[0]) / sizeof(vals[0][0]);
            std::vector<double> vals_vec;
            std::vector<double> sumw2_vec;
            vals_vec.resize(vals.size() * size);
            sumw2_vec.resize(vals.size() * size);

            Eigen::MatrixXd m;
            Eigen::MatrixXd m2;
            m.resize(vals.size(), size);
            m2.resize(vals.size(), size);
            std::cout << m.size() << std::endl;

            for (int i = 0; i < vals.size(); i++)
            {
                for (int j = 0; j < size; j++)
                {
                    m(i, j) = vals[i][j];
                    m2(i, j) = sumw2[i][j];
                }
            }
            // now use Eigen Map to flatten in C-major
            Eigen::Map<Eigen::MatrixXd>(vals_vec.data(), m.rows(), m.cols()) = m;
            Eigen::Map<Eigen::MatrixXd>(sumw2_vec.data(), m2.rows(), m2.cols()) = m2;
            return std::vector<std::vector<double>>{vals_vec, sumw2_vec};
        }
        break;
    }
}

template <typename boost_histogram, std::size_t D>
auto convertAtomics(const boost_histogram &h)
{
    if constexpr (D == 1)
    {
        std::vector<float> vals;
        std::vector<float> sumw2;
        for (auto &&x : indexed(h))
        {
            const auto n = x->value().load();
            const auto w2 = x->variance().load();
            vals.emplace_back(n);
            sumw2.emplace_back(w2);
        }
        return std::vector<std::vector<float>>{vals, sumw2};
    }
    else
    {
        Eigen::initParallel();
        std::vector<std::array<double, D>> vals;
        std::vector<std::array<double, D>> sumw2;
        for (auto &&x : indexed(h))
        {
            const auto &n = x->value(); //this is a std::array<atomic<double>,D>
            const auto &w2 = x->variance();
            std::array<double, D> n_doubles;
            std::array<double, D> w2_doubles;
            for (unsigned int j = 0; j < D; j++)
            {
                n_doubles[j] = n[j].load();
                w2_doubles[j] = w2[j].load();
            }
            vals.emplace_back(n_doubles);
            sumw2.emplace_back(w2_doubles);
        }
        int size = sizeof(vals[0]) / sizeof(vals[0][0]);
        std::vector<double> vals_vec;
        std::vector<double> sumw2_vec;
        vals_vec.resize(vals.size() * size);
        sumw2_vec.resize(vals.size() * size);

        Eigen::MatrixXd m;
        Eigen::MatrixXd m2;
        m.resize(vals.size(), size);
        m2.resize(vals.size(), size);
        std::cout << m.size() << std::endl;

        for (int i = 0; i < vals.size(); i++)
        {
            for (int j = 0; j < size; j++)
            {
                m(i, j) = vals[i][j];
                m2(i, j) = sumw2[i][j];
            }
        }
        // now use Eigen Map to flatten in C-major
        Eigen::Map<Eigen::MatrixXd>(vals_vec.data(), m.rows(), m.cols()) = m;
        Eigen::Map<Eigen::MatrixXd>(sumw2_vec.data(), m2.rows(), m2.cols()) = m2;
        return std::vector<std::vector<double>>{vals_vec, sumw2_vec};
    }
}

template <typename boost_histogram>
auto getD(const boost_histogram &h)
{
    for (auto &&x : indexed(h))
    {
        auto testVar = x->value();
        if constexpr (std::is_convertible_v<decltype(testVar), float>)
            return 1;
        else
        {
            int size = sizeof(testVar) / sizeof(testVar[0]);
            return size;
        }
    }
}

template <typename boost_histogram>
auto getRank(const boost_histogram &h)
{
    return h.rank();
}

template <typename boost_histogram>
auto getAxisSize(const boost_histogram &h, int N)
{
    return h.axis(N).size();
}

template <typename boost_histogram>
auto getAxisEdges(const boost_histogram &h, int N)
{
    auto size = getAxisSize<boost_histogram>(h, N);
    std::vector<float> axis;
    for (auto j = 0; j < size + 1; j++)
    {
        auto bin = h.axis(N).bin(j).lower();
        axis.emplace_back(bin);
    }
    return axis;
}

#endif
