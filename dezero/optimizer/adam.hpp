#ifndef __ADAM_H__
#define __ADAM_H__

#include "../optimizer.hpp"
#include <unordered_map>

using std::unordered_map;

namespace md {
	class Adam : public Optimizer {
	public:
		Adam(
			const float _alpha = 0.001f, 
			const float _beta1 = 0.9f,
			const float _beta2 = 0.999f,
			const float _eps = 1e-08f)
			: t(0.0), alpha(_alpha), beta1(_beta1), beta2(_beta2), eps(_eps) {
		
		}

		void update();
		void update_one(const parameter& param);

	protected:
		float get_lr();

	protected:
		float t;
		float alpha;
		float beta1;
		float beta2;
		float eps;
		unordered_map<ull, xarr_f> ms;
		unordered_map<ull, xarr_f> vs;
	};
}
#endif
