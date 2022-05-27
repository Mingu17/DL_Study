#include "function.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	void Function::push_ref(float f) {
		inputs.push_back(spvar::create(f));
	}

	void Function::push_ref(spvar& v) {
		inputs.push_back(v);
	}

	void Function::push_ref(const spvar& v) {
		inputs.push_back(v);
	}

	void Function::compute_outputs() {
		forward(inputs);

		if (Common::enable_backprop) {
			generation = 0;
			for (auto obj : inputs) {
				int gen = obj.get()->get_generation();
				if (gen > generation) {
					generation = gen;
				}
			}

			for (int i = 0; i < outputs.size(); ++i) {
				outputs[i].get()->set_creator(this);
			}
		}
	}
}