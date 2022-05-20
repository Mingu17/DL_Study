#ifndef __COS_H__
#define __COS_H__

#include "../function.hpp"

namespace md {
	class Cos : public Function {
	public:
		Cos() {
			param_reserve(1, 1, 1);
		}
		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	};
}
#endif
