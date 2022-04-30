#ifndef __EXP_H__
#define __EXP_H__

#include "../function.hpp"

namespace md {
	class Exp : public Function {
	public:
		Exp() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Exp class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Exp::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		xarr_d res = xt::exp(x);
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// Exp class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Exp::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& y = outputs[0];
		//		return vec_spvar({ gy * y });
		//	}
		//}
	};
}
#endif
