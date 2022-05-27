#ifndef __TANH_H__
#define __TANH_H__

#include "../function.hpp"

namespace md {
	class Tanh : public Function {
	public:
		Tanh() {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
