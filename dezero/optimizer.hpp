#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "common.hpp"
#include "model.hpp"

namespace md {
	class Optimizer {
	public:
		Optimizer() : target(nullptr) {}

		void setup(Model* _target) {
			target = _target;
		}

		virtual void update();
		virtual void update_one(const parameter& param) = 0;

	protected:
		Model* target;
		//Model& target2;
	};
}
#endif