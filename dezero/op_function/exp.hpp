#ifndef __EXP_H__
#define __EXP_H__

#include "../function.hpp"

namespace md {
	class Exp : public Function {
	public:
		Exp() {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
