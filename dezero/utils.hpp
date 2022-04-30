#ifndef __UTILS_H__
#define __UTILS_H__

#include "common.hpp"
#include <random>
#include <string>

namespace md {
	typedef xarr_d(*rand_t)(xarr_size&& s);

	class Utils {
	public:
		static xarr_d sum_to(xarr_d& x, xarr_size& shape);
		static spvar& reshape_sum_backward(
			spvar& gy, 
			xarr_size& x_shape, 
			xarr_size& axis, 
			bool keepdims
		);

		static xarr_d rand(const xarr_size& s) {
			return xt::random::rand<double>(s);
		}

		template<std::size_t L>
		static xarr_d rand(const int(&shape)[L]) {
			return xt::random::rand<double>(shape);
		}

		static xarr_d randn(const xarr_size& s) {
			return xt::random::randn<double>(s);
		}

		template<std::size_t L>
		static xarr_d randn(const int(&shape)[L]) {
			return xt::random::randn<double>(shape);
		}

		static xarr_d zeros(const xarr_size& s) {
			return xt::zeros<double>(s);
		}

		template<std::size_t L>
		static xarr_d zeros(const int(&shape)[L]) {
			return xt::zeros<double>(shape);
		}

		static xarr_d logsumexp(xarr_d& x, size_t axis);
		static xarr_d logsumexp(xarr_d& x, const xarr_size& axis);

	private:
		
	};
}
#endif
