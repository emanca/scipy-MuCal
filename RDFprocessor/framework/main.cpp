// Type your code here, or load an example.
#include <boost/histogram.hpp>
#include <memory>
#include <tuple>
#include <utility>
#include <iostream>
#include <chrono>
#include "interface/thread_safe_withvariance.hpp"

int main()
{
    auto htmp = boost::histogram::make_histogram_with(boost::histogram::dense_storage<boost::histogram::accumulators::thread_safe_withvariance<double>>(), boost::histogram::axis::regular<>(100, -1.0, 1.0));
    h(0.0, boost::histogram::weight(1));
}