#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include "../function.hpp"

namespace md {
	class Transpose : public Function {
	public:
		Transpose() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
