#ifndef __VARIABLE_H__
#define __VARIABLE_H__

#include "sp_variable.hpp"
#include <string>
#include <functional>
#include <iostream>
#include <initializer_list>
#include "local_exception.hpp"

namespace md {
	class Function;
	//class Variable;

	//typedef std::shared_ptr<Variable> spvar;
	//typedef vector<spvar> vec_spvar;

	class Variable {
	public:
		Variable(std::string _name = "") :
			creator(nullptr), generation(0), name(_name) {
			Common::xarr_init(data);
			clear_grad();
		}

		Variable(const xarr_d& _data, std::string _name = "") :
			creator(nullptr), generation(0), name(_name) {
			data = _data;
			clear_grad();
		}

		Variable(const double _data, std::string _name = "") :
			creator(nullptr), generation(0), name(_name) {
			data = xarr_d({ _data });
			clear_grad();
		}

		Variable(const Variable& v, std::string _name = "") :
			data(v.data), grad(v.grad), creator(v.creator), generation(v.generation), name(v.name) {
		}

		~Variable() {
			//std::cout << "Variable terminate." << std::endl;
			clear_grad();
		}

		xarr_d& get_data() {
			return data;
		}

		void set_data(const xarr_d& _data) {
			data = _data;
		}

		spvar& get_grad() {
			return grad;
		}

		void set_grad(spvar& _grad) {
			grad = _grad;
		}

		Function* get_creator() {
			return creator;
		}

		void set_name(std::string n) {
			name = n;
		}

		void clear_grad() {
			grad.reset();
		}

		int get_generation() {
			return generation;
		}

		const xarr_size& get_shape() {
			return data.shape();
		}

		const size_t get_ndim() {
			return data.dimension();
		}

		const size_t get_size() {
			return data.size();
		}

		const size_t get_len() {
			return data.shape()[0];
		}

		void add_grad(spvar& _grad);
		void set_creator(Function* func);
		void backward(bool retain_grad = false);

	protected:
		xarr_d data;
		spvar grad;
		Function* creator;
		int generation;
		std::string name;
	};
}

#endif
