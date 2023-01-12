// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_VEC_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_VEC_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for weighted_sum_vec<>
#include <type_traits>

namespace boost
{
    namespace histogram
    {
        namespace accumulators
        {

            /// Holds sum of weights and its variance estimate
            template <class ValueType, std::size_t D>
            class weighted_sum_vec
            {
            public:
                using value_type = ValueType;
                using vec_value_type = std::vector<value_type>;
                using const_reference = const value_type &;

                weighted_sum_vec() = default;

                /// Initialize sum to value and allow implicit conversion
                weighted_sum_vec(vec_value_type value) noexcept : weighted_sum_vec(value, value)
                {
                }

                /// Initialize sum to value and variance
                weighted_sum_vec(vec_value_type value, vec_value_type variance) noexcept
                    : sum_of_weights_(value), sum_of_weights_squared_(variance) {}

                /// Increment by one.
                weighted_sum_vec &operator++()
                {
                    std::transform(std::begin(sum_of_weights_), std::end(sum_of_weights_), std::begin(sum_of_weights_), [](int x) { return x + 1; });
                    std::transform(std::begin(sum_of_weights_squared_), std::end(sum_of_weights_squared_), std::begin(sum_of_weights_squared_), [](int x) { return x + 1; });
                    return *this;
                }

                /// Increment by weight.
                weighted_sum_vec &operator+=(const weight_type<value_type> &w)
                {
                    std::transform(std::begin(sum_of_weights_), std::end(sum_of_weights_), std::begin(sum_of_weights_), [&w](int x) { return x + w.value; });
                    std::transform(std::begin(sum_of_weights_squared_), std::end(sum_of_weights_squared_), std::begin(sum_of_weights_squared_), [&w](int x) { return x + w.value * w.value; });
                    return *this;
                }

                /// Added another weighted sum.
                weighted_sum_vec &operator+=(const weighted_sum_vec &rhs)
                {
                    for (unsigned int i = 0; i < sum_of_weights_.size(); i++)
                    {
                        sum_of_weights_[i] += rhs.sum_of_weights_[i];
                        sum_of_weights_squared_[i] += rhs.sum_of_weights_squared_[i];
                    }
                    return *this;
                }

                // bool operator==(const weighted_sum_vec &rhs) const noexcept
                // {
                //     return sum_of_weights_ == rhs.sum_of_weights_ &&
                //            sum_of_weights_squared_ == rhs.sum_of_weights_squared_;
                // }

                // bool operator!=(const weighted_sum_vec &rhs) const noexcept { return !operator==(rhs); }

                /// Return value of the sum.
                std::array<value_type, D> value() const noexcept { return sum_of_weights_; }

                /// Return estimated variance of the sum.
                std::array<value_type, D> variance() const noexcept { return sum_of_weights_squared_; }

                // lossy conversion must be explicit
                explicit operator vec_value_type() const { return sum_of_weights_; }

                /// Insert sample x.
                void operator()(vec_value_type x) { operator()(weight(1), x); }

                /// Insert sample x with weight w.
                void operator()(const weight_type<value_type> &w, vec_value_type x)
                {
                    for (unsigned int i = 0; i < x.size(); i++)
                    {
                        sum_of_weights_[i] += (w.value * x[i]);
                        sum_of_weights_squared_[i] += (w.value * w.value * x[i] * x[i]);
                    }
                }

            protected:
                std::array<value_type,D> sum_of_weights_;
                std::array<value_type,D> sum_of_weights_squared_;
            };

        } // namespace accumulators
    }     // namespace histogram
} // namespace boost

#endif