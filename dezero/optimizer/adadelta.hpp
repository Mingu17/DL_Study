#ifndef __ADA_DELTA_H__
#define __ADA_DELTA_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class AdaDelta : public Optimizer {
	public:
		AdaDelta(const float _rho = 0.95f, const float _eps = 1e-06f)
			:rho(_rho), eps(_eps) {

		}

		void update_one(const parameter& param);

	protected:
		float rho;
		float eps;
		unordered_map<ull, xarr_f> msg;
		unordered_map<ull, xarr_f> msdx;
	};
}
#endif
