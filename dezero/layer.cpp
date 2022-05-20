#include "layer.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	void Layer::push_ref(float f) {
		auto v = spvar::create(f);
		inputs.push_back(v);
	}

	void Layer::push_ref(spvar& v) {
		inputs.push_back(v);
	}

	void Layer::push_ref(const spvar& v) {
		inputs.push_back(v);
	}

	void Layer::param_add(const parameter& param, string name) {
		if (name.empty()) {
			int cnt = static_cast<int>(params.size()) - 1;
			string num_str = Common::string_format("%07d", cnt);
			params.insert(std::make_pair(param, init_name + num_str));
		}
		else {
			params.insert(std::make_pair(param, name));
		}
	}

	void Layer::clear_grad() {
		for (auto iter = params.begin(); iter != params.end(); iter++) {
			iter->first->clear_grad();
		}
	}
}