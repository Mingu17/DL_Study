#ifndef __ADAM_H__
#define __ADAM_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class Adam : public Optimizer {
	public:
		Adam(
			double _alpha = 0.001, 
			double _beta1 = 0.9, 
			double _beta2 = 0.999, 
			double _eps = 1e-08) 
			: t(0.0), alpha(_alpha), beta1(_beta1), beta2(_beta2), eps(_eps) {
		
		}

		void update();
		void update_one(const parameter& param);

	protected:
		double get_lr();

	protected:
		double t;
		double alpha;
		double beta1;
		double beta2;
		double eps;
		unordered_map<ull, xarr_d> ms;
		unordered_map<ull, xarr_d> vs;
	};
}
#endif
