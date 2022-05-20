#ifndef __LAYER_H__
#define __LAYER_H__

#include "common.hpp"
#include <set>
#include <string>

using std::set;
using std::pair;
using std::string;

namespace md {
	class Layer {
	public:
		Layer() : init_name("param") {

		}

		Layer(string _init_name) : init_name(_init_name) {

		}

		template<typename... Vs>
		vec_spvar& call(Vs&... args) {
			inputs.clear();
			push_ref(args...);
			outputs = forward(inputs);
			return outputs;
		}

		set<pair<parameter, string>>& get_params() {
			return params;
		}

		void param_add(const parameter& param, string name = "");
		void clear_grad();
		virtual vec_spvar forward(vec_spvar& xs) = 0;

	protected:
		void push_ref(float f);
		void push_ref(spvar& v);
		void push_ref(const spvar& v);
		void push_ref(void) { return; }

		template<typename V, typename... Vs>
		void push_ref(V&& var, Vs&&... vars) {
			push_ref(var);
			push_ref(vars...);
		}
	protected:
		set<pair<parameter, string>> params;
		string init_name;

		vec_spvar inputs;
		vec_spvar outputs;
	};
}
#endif