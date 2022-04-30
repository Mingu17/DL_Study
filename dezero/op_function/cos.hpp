#ifndef __COS_H__
#define __COS_H__

#include "../function.hpp"

namespace md {
	class Cos : public Function {
	public:
		Cos() {}
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
			
		///// <summary>
		///// Cos class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Cos::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		return vec_spvar({ spvar::create(xt::cos(x)) });
		//	}
		//}

		///// <summary>
		///// Cos class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Cos::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x = inputs[0];
		//		return vec_spvar({ gy * -x.sin() });// -sin(x)
		//	}
		//}
	};
}
#endif
