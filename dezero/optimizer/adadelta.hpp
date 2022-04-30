#ifndef __ADA_DELTA_H__
#define __ADA_DELTA_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class AdaDelta : public Optimizer {
	public:
		AdaDelta(double _rho = 0.95, double _eps = 1e-06)
			:rho(_rho), eps(_eps) {

		}

		void update_one(const parameter& param);

	protected:
		double rho;
		double eps;
		unordered_map<ull, xarr_d> msg;
		unordered_map<ull, xarr_d> msdx;
	};
}
#endif
