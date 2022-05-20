#ifndef __SGD_H__
#define __SGD_H__

#include "../optimizer.hpp"

namespace md {
	class SGD : public Optimizer {
	public:
		SGD(const float _lr = 0.01f) : lr(_lr) {}

		void update_one(const parameter& param);

	protected:
		float lr;
	};
}

#endif