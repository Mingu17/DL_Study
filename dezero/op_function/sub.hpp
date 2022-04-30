#ifndef __SUB_H__
#define __SUB_H__

#include "../function.hpp"

namespace md {
	class Sub : public Function {
	public:
		Sub() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Sub class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 2) {
		//		throw LocalException("(Sub::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x0 = xs[0]->get_data();
		//		xarr_d& x1 = xs[1]->get_data();
		//		return vec_spvar({ spvar::create(x0 - x1) });
		//	}
		//}


		///// <summary>
		///// Sub class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Sub::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		return vec_spvar({ gy, -gy });
		//	}
		//}
	};
}
#endif
