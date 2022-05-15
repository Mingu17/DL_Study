#ifndef __COMMON_H__
#define __COMMON_H__

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <vector>
#include <memory>
#include "common_const.hpp"

using std::vector;

namespace md {
	class spvar;
	using namespace xt::placeholders;

	typedef xt::xarray<double> xarr_d;
	typedef xt::xarray<float> xarr_f;
	typedef xt::xarray<int> xarr_i;
	typedef xt::xarray<unsigned char> xarr_uc;
	typedef xt::xarray<char> xarr_c;
	typedef xt::svector<size_t> xarr_size;
	typedef vector<spvar> vec_spvar;
	typedef vector<xarr_d> vec_xarr_d;
	typedef spvar parameter;
	typedef unsigned long long ull;
	typedef xt::xdynamic_slice_vector vec_xslice;
	
	template<class T>
	using SP = std::shared_ptr<T>;

	class Common {
	public:
		static bool xarr_isinit(const xarr_d& arr) {
			if (arr.size() == 1 && arr[0] == DBL_MAX) {
				return false;
			}
			return true;
		}

		static bool isNone(const xarr_d& arr) {
			if (arr.size() == 1 && arr[0] == DBL_MAX) {
				return true;
			}
			return false;
		}

		//static void xarr_init(xarr_d& arr) {
		//	arr = xarr_d({ DBL_MAX });
		//}

		template<typename ... Args>
		static std::string string_format(const std::string& format, Args ... args)
		{
			int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
			if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
			auto size = static_cast<size_t>(size_s);
			std::unique_ptr<char[]> buf(new char[size]);
			std::snprintf(buf.get(), size, format.c_str(), args ...);
			return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
		}

		static void no_grad() {
			enable_backprop = false;
		}

		static bool enable_backprop;
	};

	template<class T, class... Args>
	static SP<T> SPK(Args... args) {
		return std::make_shared<T>(std::forward<Args>(args)...);
	}
}
#endif
