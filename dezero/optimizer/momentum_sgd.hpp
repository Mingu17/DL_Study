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
		MomentumSGD(const float _lr = 0.01f, const float _momentum = 0.9f) :
			lr(_lr), momentum(_momentum) {}

		void update_one(const parameter& param);

	protected:
		float lr;
		float momentum;
		unordered_map<ull, xarr_f> vs;
	};
}
#endif
