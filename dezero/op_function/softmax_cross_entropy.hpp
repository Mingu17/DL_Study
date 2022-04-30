#ifndef __SOFTMAX_CROSS_ENTROPY_H__
#define __SOFTMAX_CROSS_ENTROPY_H__

#include "../function.hpp"

namespace md {
	class SoftmaxCrossEntropy : public Function {
	public:
		SoftmaxCrossEntropy() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
	};
}
#endif