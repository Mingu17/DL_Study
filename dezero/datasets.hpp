#ifndef __DATA_SET_H__
#define __DATA_SET_H__

#include "transforms.hpp"
#include <utility>
using std::pair;

namespace md {
	class Dataset {
	public:
		Dataset(const bool _train = false,
			const vector<SP<Transforms>>& _transforms = {},
			const vector<SP<Transforms>>& _target_transforms = {})
			: train(_train), transforms(_transforms), target_transforms(_target_transforms) {
			//call prepare
		}

		~Dataset() {
			transforms.clear();
			target_transforms.clear();
		}

		size_t get_len() const {
			if (train_data.size() > 0) {
				return train_data.shape()[0];
			}
			else return 0;
		}

		pair<xarr_f, float> operator[](const size_t index) const {
			size_t t_size = train_data.size();
			if (t_size > 0 && index < t_size) {
				float label = train_label(index);
				return std::make_pair(xt::view(train_data, index), label);
			}
			else {
				THROW_EXCEPTION("index is out of range");
			}
		}

		pair<xarr_f, int> operator()(const size_t index) const {
			size_t t_size = train_data.size();
			if (t_size > 0 && index < t_size) {
				int label = static_cast<int>(train_label(index));
				return std::make_pair(xt::view(train_data, index), label);
			}
			else {
				THROW_EXCEPTION("index is out of range");
			}
		}

		xarr_f get_train_data(const xarr_i& batch_index) const {
			if (train_data.size() > 0) {
				return xt::view(train_data, xt::keep(batch_index));
			}
			else {
				THROW_EXCEPTION("train data is not exist");
			}
		}

		xarr_f get_train_label(const xarr_i& batch_index) const {
			if (train_label.size() > 0) {
				return xt::view(train_label, xt::keep(batch_index));
			}
			else {
				THROW_EXCEPTION("train label is not exist");
			}
		}

		virtual void prepare() = 0;

	protected:
		void transform_data(const xarr_f& in_data, const xarr_f& in_label) {
			if (transforms.size() > 0) {
				xarr_f t_data = transforms[0]->compute(in_data);
				for (size_t i = 1; i < transforms.size(); ++i) {
					t_data = transforms[i]->compute(t_data);
				}
				train_data = t_data;
			}
			else {
				train_data = in_data;
			}

			if (target_transforms.size() > 0) {
				xarr_f t_label = target_transforms[0]->compute(in_label);
				for (size_t i = 1; i < target_transforms.size(); ++i) {
					t_label = target_transforms[i]->compute(t_label);
				}
				train_label = t_label;
			}
			else {
				train_label = in_label;
			}
		}
	protected:
		bool train;
		vector<SP<Transforms>> transforms;
		vector<SP<Transforms>> target_transforms;
		xarr_f train_data;
		xarr_f train_label;
	};
}

#endif
