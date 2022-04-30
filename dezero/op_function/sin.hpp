#ifndef __SIN_H__
#define __SIN_H__

#include "../function.hpp"

namespace md {
	class Sin : public Function {
	public:
		Sin() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Sin class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Sin::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		return vec_spvar({ spvar::create(xt::sin(x)) });
		//	}
		//}

		///// <summary>
		///// Sin class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Sin::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x = inputs[0];
		//		return vec_spvar({ gy * x.cos() }); //cos(x) });
		//	}
		//}
	};
}
#endif