#ifndef __SOFTMAX_CROSS_ENTROPY_H__
#define __SOFTMAX_CROSS_ENTROPY_H__

#include "../function.hpp"

namespace md {
	class SoftmaxCrossEntropy : public Function {
	public:
		SoftmaxCrossEntropy() {
			param_reserve(2, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
