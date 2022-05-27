#ifndef __POW_H__
#define __POW_H__

#include "../function.hpp"

namespace md {
	class Pow : public Function {
	public:
		Pow(const float _c) : c(_c) {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	protected:
		float c;
	};
}
#endif
