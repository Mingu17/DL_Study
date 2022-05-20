#ifndef __DIV_H__
#define __DIV_H__

#include "../function.hpp"

namespace md {
	class Div : public Function {
	public:
		Div() {
			param_reserve(2, 1, 2);
		}

		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
