#include "optimizer.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"
#include <set>
#include <string>

namespace md {
	void Optimizer::update() {
		std::set<std::pair<parameter, std::string>>& params = target->get_params();
		for (auto iter = params.begin(); iter != params.end(); iter++) {
			update_one(iter->first);
		}
	}
}