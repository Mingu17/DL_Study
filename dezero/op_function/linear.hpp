#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../function.hpp"

namespace md {
	class Linear : public Function {
	public:
		Linear() {
			param_reserve(3, 1, 3);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
