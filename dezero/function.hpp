#ifndef __FUNCTION_H__
#define __FUNCTION_H__

//#include "sp_variable.hpp"
#include "common.hpp"
#include <cstdarg>
#include "local_exception.hpp"

namespace md {
	class Function {
	public:
		Function() : prev_func(nullptr), generation(0) {}
		~Function() {
			//std::cout << "Function Terminate." << std::endl;
		}

		Function(Function* func) : prev_func(func), generation(0) {}

		template<typename... Vs>
		vec_spvar& call(Vs&... args) {
			inputs.clear();
			push_ref(args...);
			compute_outputs();
			return outputs;
		}

		vec_spvar& call(const vec_spvar& _input) {
			inputs.clear();
			for (int i = 0; i < _input.size(); ++i) {
				inputs.push_back(_input[i]);
			}
			compute_outputs();

			return outputs;
		}

		virtual void forward(const vec_spvar& xs) = 0;
		virtual void backward(const vec_spvar& gys) = 0;

		void param_reserve(size_t in, size_t out, size_t g) {
			inputs.reserve(in);
			outputs.resize(out);
			grads.reserve(g);
		}

		Function& operator()(Function& func) {
			prev_func = &func;
			return *this;
		}

		vec_spvar& get_inputs() {
			return inputs;
		}

		vec_spvar& get_outputs() {
			return outputs;
		}

		vec_spvar& get_grads() {
			return grads;
		}

		int get_generation() {
			return generation;
		}

	protected:	
		void push_ref(float f);
		void push_ref(spvar& v);
		void push_ref(const spvar& v);
		void push_ref(void) { return; }
	
		template<typename V, typename... Vs>
		void push_ref(V&& var, Vs&&...vars) {
			push_ref(var);
			push_ref(vars...);
		}
	
		void compute_outputs();
	protected:
		Function* prev_func;
		vec_spvar inputs;
		vec_spvar outputs;
		vec_spvar grads;
		int generation;
	};
}
#endif
