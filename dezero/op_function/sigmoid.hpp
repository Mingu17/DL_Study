#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "../function.hpp"

namespace md {
	class Sigmoid : public Function {
	public:
		Sigmoid() {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
