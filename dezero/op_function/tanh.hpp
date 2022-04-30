#ifndef __TANH_H__
#define __TANH_H__

#include "../function.hpp"

namespace md {
	class Tanh : public Function {
	public:
		Tanh() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// Tanh class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Tanh::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		return vec_spvar({ spvar::create(xt::tanh(x)) });
		//	}
		//}

		///// <summary>
		///// Tanh class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Tanh::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& y = outputs[0];
		//		return vec_spvar({ gy * (1.0 - y * y) }); // vpow(y, 2) ?
		//	}
		//}
	};
}
#endif
