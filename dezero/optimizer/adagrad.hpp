#ifndef _ADA_GRAD_H__
#define _ADA_GRAD_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class AdaGrad : public Optimizer {
	public:
		AdaGrad(const float _lr = 0.001f, const float _eps = 1e-08f)
			:lr(_lr), eps(_eps) {

		}

		void update_one(const parameter& param);
	protected:
		float lr;
		float eps;
		unordered_map<ull, xarr_f> hs;
	};
}
#endif