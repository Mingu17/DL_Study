#include "model.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	vec_spvar& Model::get_params() {
		if (params.size() == 0) {
			for (auto l_iter = layers.begin(); l_iter != layers.end(); l_iter++) {
				auto& l_params = (*l_iter)->get_params();
				for (auto p_iter = l_params.begin(); p_iter != l_params.end(); p_iter++) {
					params.push_back(*p_iter);
				}
			}
		}
		return params;
	}
}