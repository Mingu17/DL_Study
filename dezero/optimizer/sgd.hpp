#ifndef __SGD_H__
#define __SGD_H__

#include "../optimizer.hpp"

namespace md {
	class SGD : public Optimizer {
	public:
		SGD(double _lr = 0.01) : lr(_lr) {}

		void update_one(const parameter& param);

	protected:
		double lr;
	};
}

#endif