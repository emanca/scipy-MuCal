// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_WITHVARIANCE_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_WITHVARIANCE_HPP

#include <atomic>
#include <boost/core/nvp.hpp>
#include <boost/mp11/utility.hpp>
#include <type_traits>
#include <vector>
#include <boost/histogram/fwd.hpp> // for weighted_sum<>

namespace boost
{
    namespace histogram
    {
        namespace accumulators
        {

            template <class T>
            class thread_safe_withvariance
            {
            public:
                using value_type = T;
                using const_reference = const value_type &;
                using super_t = std::atomic<T>;
                using const_reference_atomic = const super_t &;

                // // initialize to 0
                // thread_safe_withvariance() noexcept : thread_safe_withvariance(super_t(static_cast<T>(0)), super_t(static_cast<T>(0))) {}
                // non-atomic copy and assign is allowed, because storage is locked in this case
                thread_safe_withvariance() = default;
                // copy constructor
                thread_safe_withvariance(const thread_safe_withvariance &o) noexcept : sum_of_weights_(o.value().load()), sum_of_weights_squared_(o.variance().load()) {}

                thread_safe_withvariance &operator=(const thread_safe_withvariance &o) noexcept
                {
                    sum_of_weights_.store(o.value().load());
                    sum_of_weights_squared_.store(o.variance().load());
                    return *this;
                }
                // initialize to value and variance
                thread_safe_withvariance(const_reference value, const_reference variance) : sum_of_weights_(value), sum_of_weights_squared_(variance) {}
                // initialize to weight type
                template <class weightType>
                thread_safe_withvariance &operator=(const weight_type<weightType> &w)
                {
                    sum_of_weights_.store(w.value.load());
                    sum_of_weights_squared_.store(w.variance.load());
                    return *this;
                }
                // add two thread safe objects
                thread_safe_withvariance &operator+=(const thread_safe_withvariance &arg)
                {
                    sum_of_weights_ += arg.sum_of_weights_.load();
                    sum_of_weights_squared_ += arg.sum_of_weights_squared_.load();
                    return *this;
                }

                thread_safe_withvariance &operator+=(value_type arg)
                {
                    // use Josh's trick for summing doubles
                    double old_val = sum_of_weights_.load();
                    double desired_val = old_val + arg;
                    while (!sum_of_weights_.compare_exchange_weak(old_val, desired_val))
                    {
                        desired_val = old_val + arg;
                    }

                    double old_variance = sum_of_weights_squared_.load();
                    double desired_variance = old_variance + arg * arg;
                    while (!sum_of_weights_squared_.compare_exchange_weak(old_variance, desired_variance))
                    {
                        desired_variance = old_variance + arg * arg;
                    }
                    
                    return *this;
                }
                // Increment by one.
                thread_safe_withvariance &operator++()
                {
                    // use Josh's trick for summing doubles
                    double old_val = sum_of_weights_.load();
                    double desired_val = old_val + 1.;
                    while (!sum_of_weights_.compare_exchange_weak(old_val, desired_val))
                    {
                        desired_val = old_val + 1.;
                    }

                    double old_variance = sum_of_weights_squared_.load();
                    double desired_variance = old_variance + 1. * 1.;
                    while (!sum_of_weights_squared_.compare_exchange_weak(old_variance, desired_variance))
                    {
                        desired_variance = old_variance + 1. * 1.;
                    }
                    // update the current object
                    sum_of_weights_.store(desired_val);
                    sum_of_weights_squared_.store(desired_variance);
                    return *this;
                }
                // add a basic type to thread safe objects
                template <class weightType>
                thread_safe_withvariance &operator+=(const weight_type<weightType> &w)
                {
                    // use Josh's trick for summing doubles
                    double old_val = sum_of_weights_.load();
                    double desired_val = old_val + w.value;
                    while (!sum_of_weights_.compare_exchange_weak(old_val, desired_val))
                    {
                        desired_val = old_val + w.value;
                    }

                    double old_variance = sum_of_weights_squared_.load();
                    double desired_variance = old_variance + w.value * w.value;
                    while (!sum_of_weights_squared_.compare_exchange_weak(old_variance, desired_variance))
                    {
                        desired_variance = old_variance + w.value * w.value;
                    }
                    // update the current object
                    sum_of_weights_.store(desired_val);
                    sum_of_weights_squared_.store(desired_variance);
                    return *this;
                }

                bool operator==(const thread_safe_withvariance &rhs) const noexcept
                {
                    return sum_of_weights_ == rhs.sum_of_weights_ &&
                           sum_of_weights_squared_ == rhs.sum_of_weights_squared_;
                }

                bool operator!=(const thread_safe_withvariance &rhs) const noexcept { return !operator==(rhs); }

                /// Return value of the sum.
                const_reference_atomic value() const noexcept { return sum_of_weights_; }

                /// Return estimated variance of the sum.
                const_reference_atomic variance() const noexcept { return sum_of_weights_squared_; }

                // lossy conversion must be explicit
                explicit operator const_reference() const { return sum_of_weights_; }

                template <class Archive>
                void serialize(Archive &ar, unsigned /* version */)
                {
                    ar &make_nvp("sum_of_weights", sum_of_weights_);
                    ar &make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
                }

            private:
                super_t sum_of_weights_ = super_t(static_cast<value_type>(0));
                super_t sum_of_weights_squared_ = super_t(static_cast<value_type>(0));
            };

        } // namespace accumulators
    }     // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std
{
    template <class T, class U>
    struct common_type<boost::histogram::accumulators::thread_safe_withvariance<T>,
                       boost::histogram::accumulators::thread_safe_withvariance<U>>
    {
        using type = boost::histogram::accumulators::thread_safe_withvariance<common_type_t<T, U>>;
    };

    template <class T, class U>
    struct common_type<boost::histogram::accumulators::thread_safe_withvariance<T>, U>
    {
        using type = boost::histogram::accumulators::thread_safe_withvariance<common_type_t<T, U>>;
    };

    template <class T, class U>
    struct common_type<T, boost::histogram::accumulators::thread_safe_withvariance<U>>
    {
        using type = boost::histogram::accumulators::thread_safe_withvariance<common_type_t<T, U>>;
    };
} // namespace std
#endif

#endif