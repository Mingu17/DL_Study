#ifndef __MOMENTUM_SGD_H__
#define __MOMENTUM_SGD_H__

#include "../optimizer.hpp"
#include <unordered_map>
#include <string>

using std::unordered_map;
using std::string;

namespace md {
	class MomentumSGD : public Optimizer {
	public:
		MomentumSGD(double _lr = 0.01, double _momentum = 0.9) :
			lr(_lr), momentum(_momentum) {}

		void update_one(const parameter& param);

	protected:
		double lr;
		double momentum;
		unordered_map<ull, xarr_d> vs;
	};
}
#endif
