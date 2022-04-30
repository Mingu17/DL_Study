#ifndef __MUL_H__
#define __MUL_H__

#include "../function.hpp"

namespace md {
	class Mul : public Function {
	public:
		Mul() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// Mul class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 2) {
		//		throw LocalException("(Mul::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x0 = xs[0]->get_data();
		//		xarr_d& x1 = xs[1]->get_data();
		//		xarr_d res = x0 * x1;
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// Mul class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Mul::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x0 = inputs[0];
		//		spvar& x1 = inputs[1];
		//		return vec_spvar({ gy * x1, gy * x0 });
		//	}
		//}
	};
}
#endif
