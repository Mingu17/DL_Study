#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "../function.hpp"

namespace md {
	class Softmax : public Function {
	public:
		Softmax(size_t _axis = 1) {
			axis = xarr_size({ _axis });
		}

		Softmax(const xarr_size& _axis) : axis(_axis) {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

	protected:
		xarr_size axis;
	};
}
#endif
