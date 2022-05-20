#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include "../function.hpp"

namespace md {
	class Transpose : public Function {
	public:
		Transpose() {
			param_reserve(1, 1, 1);
		}

		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
