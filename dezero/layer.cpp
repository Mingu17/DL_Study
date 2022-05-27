#include "layer.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	void Layer::push_ref(float f) {
		//auto v = spvar::create(f);
		inputs.push_back(spvar::create(f));
	}

	void Layer::push_ref(spvar& v) {
		inputs.push_back(v);
	}

	void Layer::push_ref(const spvar& v) {
		inputs.push_back(v);
	}

	void Layer::clear_grad() {
		for (auto iter = params.begin(); iter != params.end(); iter++) {
			(*iter)->clear_grad();
		}
	}
}