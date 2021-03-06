#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "../function.hpp"

namespace md {
	class MatMul : public Function {
	public:
		MatMul() {
			param_reserve(2, 1, 2);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
