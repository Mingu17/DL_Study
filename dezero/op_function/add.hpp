#ifndef __ADD_H__
#define __ADD_H__

#include "../function.hpp"

namespace md {
	class Add : public Function {
	public:
		Add() {}
		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);

	protected:
		xarr_size x0_shape;
		xarr_size x1_shape;
	};
}
#endif