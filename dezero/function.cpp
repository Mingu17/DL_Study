#include "function.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	void Function::push_ref(double d) {
		//auto v = spvar::spvar_create(d);
		auto v = spvar::create(d);
		inputs.push_back(v);
	}

	void Function::push_ref(spvar& v) {
		inputs.push_back(v);
	}

	void Function::push_ref(const spvar& v) {
		inputs.push_back(v);
	}

	void Function::compute_outputs() {
		//std::cout << "======= " << typeid(*this).name() << " =======" << std::endl;
		//std::cout << "inputs : ";
		//for (int i = 0; i < inputs.size(); ++i) {
		//	std::cout << inputs[i] << ", ";
		//}
		//std::cout << std::endl;
		outputs = forward(inputs);
		//std::cout << "outputs : " << outputs[0] << std::endl;

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