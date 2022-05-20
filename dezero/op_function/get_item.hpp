#ifndef __GET_ITEM_H__
#define __GET_ITEM_H__

#include "../function.hpp"

namespace md {
	class GetItem : public Function {
	public:
		GetItem(const vec_xslice& _slices) : slices(_slices) {
			param_reserve(1, 1, 1);
		}

		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);

	protected:
		vec_xslice slices;
	};
}
#endif