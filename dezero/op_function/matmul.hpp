#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "../function.hpp"

namespace md {
	class MatMul : public Function {
	public:
		MatMul() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
