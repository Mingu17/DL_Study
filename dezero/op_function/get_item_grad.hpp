#ifndef __GET_ITEM_GRAD_H__
#define __GET_ITEM_GRAD_H__

#include "../function.hpp"

namespace md {
	class GetItemGrad : public Function {
	public:
		GetItemGrad(const vec_xslice& _slices, const xarr_size& _in_shape)
			: slices(_slices), in_shape(_in_shape) {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

	protected:
		vec_xslice slices;
		xarr_size in_shape;
	};
}
#endif
