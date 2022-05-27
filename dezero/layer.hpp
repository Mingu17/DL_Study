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
			forward(inputs);
			return outputs;
		}

		virtual vec_spvar& get_params() {
			return params;
		}

		void clear_grad();
		virtual void forward(vec_spvar& xs) = 0;

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
		//set<pair<parameter, string>> params;
		vec_spvar params;
		string init_name;

		vec_spvar inputs;
		vec_spvar outputs;
	};
}
#endif