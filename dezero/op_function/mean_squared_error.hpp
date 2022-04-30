#ifndef __MEAN_SQUARED_ERROR_H__
#define __MEAN_SQUARED_ERROR_H__

#include "../function.hpp"

namespace md {
	class MeanSquaredError : public Function {
	public:
		MeanSquaredError() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// MeanSquaredError class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 2) {
		//		throw LocalException("(MeanSquaredError::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x0 = xs[0]->get_data();
		//		xarr_d& x1 = xs[1]->get_data();
		//		xarr_d diff = xt::pow(x0 - x1, 2);
		//		xarr_d res = xt::sum(diff) / diff.size();
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// MeanSquaredError class forward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(MeanSquaredError::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x0 = inputs[0];
		//		spvar& x1 = inputs[1];
		//		spvar diff = x0 - x1;
		//		spvar gx0 = gy * diff * (2.0 / diff->get_size());
		//		spvar gx1 = -gx0;
		//		return vec_spvar({ gx0, gx1 });
		//	}
		//}
	};
}
#endif
