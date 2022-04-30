#ifndef _ADA_GRAD_H__
#define _ADA_GRAD_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class AdaGrad : public Optimizer {
	public:
		AdaGrad(double _lr = 0.001, double _eps = 1e-08)
			:lr(_lr), eps(_eps) {

		}

		void update_one(const parameter& param);
	protected:
		double lr;
		double eps;
		unordered_map<ull, xarr_d> hs;
	};
}
#endif