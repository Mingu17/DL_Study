#ifndef __SP_VARIABLE_H__
#define __SP_VARIABLE_H__

#include "common.hpp"
#include "local_exception.hpp"
#include <string>
//#include "variable.hpp"

namespace md {
	class Variable;
	class spvar_count {
	public:
		spvar_count() : cnt(nullptr) {}
		spvar_count(const spvar_count& _cnt) : cnt(_cnt.cnt) {}

		void swap(spvar_count& _cnt) {
			std::swap(cnt, _cnt.cnt);
		}

		unsigned int use_count() const {
			if (cnt != nullptr) {
				return *cnt;
			}
			else return 0;
		}

		template<class T>
		void add(T* p) {
			if (p != nullptr) {
				if (cnt == nullptr) {
					cnt = new unsigned int(1);
				}
				else {
					++(*cnt);
				}
			}
		}

		template<class T>
		void release(T* p) {
			if (cnt != nullptr) {
				--(*cnt);
				if (*cnt == 0) {
					delete p;
					delete cnt;
				}
				cnt = nullptr;
			}
		}
	protected:
		unsigned int* cnt;
	};

	class spvar {
	public:
		spvar() : ptr(nullptr), cnt() { setup_id(); }

		explicit spvar(Variable* p) noexcept : cnt() {
			add(p);
			setup_id();
		}

		spvar(const spvar& _ptr, Variable* p) : cnt(_ptr.cnt), id(_ptr.id) {
			add(p);
		}

		spvar(const spvar& _ptr) noexcept : cnt(_ptr.cnt), id(_ptr.id) {
			//if (_ptr.ptr == nullptr || _ptr.cnt.use_count() == 0) {
			//	throw LocalException("(spvar::spvar) - Source pointer is not correct");
			//}
			//else {
			//	add(_ptr.ptr);
			//}
			add(_ptr.ptr);
		}

		spvar(spvar&& _ptr) noexcept : cnt(std::move(_ptr.cnt)), id(std::move(_ptr.id)) {
			//if (_ptr.ptr == nullptr || _ptr.cnt.use_count() == 0) {
			//	throw LocalException("(spvar::spvar) - Source pointer is not correct");
			//}
			//else {
			//	add(std::move(_ptr.ptr));
			//}
			add(std::move(_ptr.ptr));
		}

		spvar& operator=(const spvar& _ptr) noexcept {
			spvar(_ptr).swap(*this);
			return *this;
		}

		spvar& operator=(spvar&& _ptr) noexcept {
			spvar(std::move(_ptr)).swap(*this);
			return *this;
		}

		~spvar() {
			release();
		}

		void reset() {
			release();
		}

		void reset(Variable* p) {
			if (p == nullptr || ptr != p) {
				throw LocalException("(spvar::reset) - Source pointer (Variable) is not correct");
			}
			else {
				release();
				add(p);
			}
		}

		void swap(spvar& _ptr) {
			std::swap(ptr, _ptr.ptr);
			cnt.swap(_ptr.cnt);
			std::swap(id, _ptr.id);
		}

		explicit operator bool() const {
			return get() != nullptr;
		}

		bool unique() const {
			return (cnt.use_count() == 1);
		}

		unsigned int use_count() const {
			return cnt.use_count();
		}

		Variable& operator*() const {
			if (ptr == nullptr) {
				throw LocalException("(spvar::operator*) - Pointer is null");
			}
			else {
				return *ptr;
			}
		}

		Variable* operator->() const {
			if (ptr == nullptr) {
				throw LocalException("(spvar::operator->) - Pointer is null");
			}
			else {
				return ptr;
			}
		}

		Variable* get() const {
			return ptr;
		}

		ull get_id() const {
			return id;
		}

		//operator

		bool operator==(const spvar& _ptr) const {
			return (ptr == _ptr.get());
		}

		bool operator==(nullptr_t) const {
			return (ptr == nullptr);
		}

		friend bool operator==(nullptr_t, const spvar& v) {
			return (nullptr == v.get());
		}

		bool operator!=(const spvar& _ptr) const {
			return (ptr != _ptr.get());
		}

		bool operator!=(nullptr_t) const {
			return (ptr != nullptr);
		}

		friend bool operator!=(nullptr_t, const spvar& v) {
			return (nullptr != v.get());
		}

		bool operator<=(const spvar& _ptr) const{
			return (ptr <= _ptr.get());
		}

		bool operator<(const spvar& _ptr) const {
			return (ptr < _ptr.get());
		}

		bool operator>=(const spvar& _ptr) const {
			return (ptr >= _ptr.get());
		}

		bool operator>(const spvar& _ptr) const {
			return (ptr > _ptr.get());
		}

		friend std::ostream& operator<<(std::ostream& out, spvar& v);

		spvar& operator+(const spvar& x);
		spvar& operator+(const double& x);
		friend spvar& operator+(const double& x0, const spvar& x1);

		spvar& operator*(const spvar& x);
		spvar& operator*(const double& x);
		friend spvar& operator*(const double& x0, const spvar& x1);

		spvar& operator-(const spvar& x);
		spvar& operator-(const double& x);
		friend spvar& operator-(const double& x0, const spvar& x1);

		spvar& operator-();

		spvar& operator/(const spvar& x);
		spvar& operator/(const double& x);
		friend spvar& operator/(const double& x0, const spvar& x1);


		spvar& pow(const double c);
		spvar& sin();
		spvar& cos();
		spvar& tanh();
		spvar& exp();
		spvar& reshape(xarr_size target_size);
		spvar& transpose();
		spvar& T();

		spvar& broadcast_to(xarr_size& shape);
		spvar& sum_to(xarr_size& shape);
		spvar& sum(const xarr_size& axis = xarr_size(), bool keepdims = false);
		spvar& dot(spvar& W);
		spvar& matmul(spvar& x); //equal dot()
		spvar& mean_squared_error(spvar& x0);
		spvar& softmax(size_t axis = 1);
		spvar& softmax(const xarr_size& axis);
		spvar& softmax_cross_entropy(spvar& t);
		spvar& relu();

		spvar& linear(spvar& W, spvar& b);
		spvar& linear(spvar& W, spvar&& b = spvar());
		spvar& linear_simple(spvar& W, spvar&& b = spvar());

		spvar& sigmoid_simple();
		spvar& sigmoid();

		spvar& get_item(const vec_xslice& slices);
		spvar& clip(double x_min, double x_max);

		static spvar create(const double& in);
		static spvar create(const xarr_d& in);
		static spvar create(const Variable& in);

	private:		
		void add(Variable* p) {
			cnt.add(p);
			ptr = p;
		}

		void release() {
			cnt.release(ptr);
			ptr = nullptr;
		}

		void setup_id();

		friend class spvar;

	private:
		Variable* ptr;
		spvar_count cnt;
		//std::string id;
		ull id;
	};
}
#endif
