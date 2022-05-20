#ifndef __RESHAPE_H__
#define __RESHAPE_H__

#include "../function.hpp"

namespace md {
	class Reshape : public Function {
	public:
		Reshape(const xarr_size& s) : shape(s) {
			param_reserve(1, 1, 1);
		}
		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
