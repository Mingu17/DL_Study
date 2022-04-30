#ifndef __POW_H__
#define __POW_H__

#include "../function.hpp"

namespace md {
	class Pow : public Function {
	public:
		Pow(double _c);
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// Pow class constructor (Function)
		///// </summary>
		///// <param name="_c"> - quotient</param>
		//Pow(double _c) : c(_c) {

		//}

		///// <summary>
		///// Pow class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	//std::cout << "c : " << c << std::endl;
		//	if (xs.size() != 1) {
		//		throw LocalException("(Pow::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		return vec_spvar({ spvar::create(xt::pow(x, c)) });
		//	}
		//}

		///// <summary>
		///// Pow class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Pow::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x = inputs[0];
		//		return vec_spvar({ c * x.pow(c - 1) * gy });
		//	}
		//}

	protected:
		double c;
	};
}
#endif
