#include "variable.hpp"
#include "function_set.hpp"
#include <iostream>
#include <queue>
#include <set>

using std::set;
using std::priority_queue;

namespace md {

	void Variable::backward(bool retain_grad) {
		if (grad.get() == nullptr) {
			grad = spvar::create(xt::ones_like(data));
		}

		set<Function*> seen_set;

		auto compare = [](Function* f0, Function* f1) {
			return f0->get_generation() < f1->get_generation();
		};
		priority_queue<Function*, vector<Function*>, decltype(compare)> funcs(compare);

		auto add_func = [&](Function* f) {
			if (seen_set.find(f) == seen_set.end()) {
				funcs.push(f);
				seen_set.insert(f);
			}
		};

		add_func(creator);

		while (!funcs.empty()) {
			Function* func = funcs.top();
			vec_spvar& gys = func->get_outputs();

			funcs.pop();

			if (Common::enable_backprop) {
				vec_spvar gxs = func->backward(gys);
				vec_spvar& inputs = func->get_inputs();

				for (int i = 0, j = 0; i < inputs.size() && j < gxs.size(); ++i, ++j) {
					Variable* x = inputs[i].get();

					if (x->get_grad().get() == nullptr) {
						x->set_grad(gxs[i]);
					}
					else {
						x->add_grad(gxs[i]);
					}

					Function* x_creator = x->get_creator();

					if (x_creator != nullptr) {
						add_func(x_creator);
					}
				}
			}

			if (!retain_grad) {
				for (auto gy : gys) {
					gy.get()->clear_grad();
				}
			}
		}
	}

	void Variable::set_creator(Function* func) {
		creator = func;
		generation = func->get_generation() + 1;
	}

	void Variable::add_grad(spvar& _grad) {
		if (grad.get()->get_size() != _grad.get()->get_size()) {
			throw LocalException("(Variable::add_grad) - Size mismatch");
		}
		grad = grad + _grad;
	}
}