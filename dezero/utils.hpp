#ifndef __UTILS_H__
#define __UTILS_H__

#include "common.hpp"
#include <random>
#include <string>

namespace md {
	//typedef xarr_d(*rand_t)(xarr_size&& s);

	class Utils {
	public:
		static xarr_f sum_to(const xarr_f& x, const xarr_size& shape);
		static spvar& reshape_sum_backward(
			spvar& gy, 
			const xarr_size& x_shape, 
			const xarr_size& axis, 
			bool keepdims
		);

		static xarr_f rand(const xarr_size& s) {
			return xt::random::rand<float>(s);
		}

		template<std::size_t L>
		static xarr_f rand(const int(&shape)[L]) {
			return xt::random::rand<float>(shape);
		}

		static xarr_f randn(const xarr_size& s) {
			return xt::random::randn<float>(s);
		}

		template<std::size_t L>
		static xarr_f randn(const int(&shape)[L]) {
			return xt::random::randn<float>(shape);
		}

		static xarr_f zeros(const xarr_size& s) {
			return xt::zeros<float>(s);
		}

		template<std::size_t L>
		static xarr_f zeros(const int(&shape)[L]) {
			return xt::zeros<float>(shape);
		}

		static xarr_f logsumexp(const xarr_f& x, size_t axis);
		static xarr_f logsumexp(const xarr_f& x, const xarr_size& axis);

		static void get_spiral(vec_xarr_f& out_data, bool train = true);
		static void get_spiral(xarr_f& out_data, xarr_f& out_label, bool train = true);
	private:
		
	};
}
#endif
