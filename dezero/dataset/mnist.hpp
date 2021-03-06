#ifndef __MNIST_H__
#define __MNIST_H__

#include "../datasets.hpp"
#include "../transform/flatten.hpp"
#include "../transform/normalize.hpp"
#include <fstream>
#include <string>

using std::string;

namespace md {
	class MNIST : public Dataset {
	public:
		MNIST(const bool _train = true,
			const vector<SP<Transforms>>& _transforms = { SPK<Flatten>(), SPK<Normalize>(0.0f, 255.0f) },
			const vector<SP<Transforms>>& _target_transforms = {},
			string _data_path = "", string _label_path = "")
			: Dataset(_train, _transforms, _target_transforms), data_path(_data_path), label_path(_label_path) {
			prepare();
		}

		void prepare() {
			string target, label;
			if (data_path.empty() && label_path.empty()) {
				if (train) {
					target = "D:\\TrainData\\MNIST\\train-images.idx3-ubyte"; //user define
					label = "D:\\TrainData\\MNIST\\train-labels.idx1-ubyte";
				}
				else {
					target = "D:\\TrainData\\MNIST\\t10k-images.idx3-ubyte";
					label = "D:\\TrainData\\MNIST\\t10k-labels.idx1-ubyte";
				}
			}
			else if (!data_path.empty() && !label_path.empty()) {
				target = data_path;
				label = label_path;
			}
			else if (!data_path.empty() && label_path.empty()) {
				THROW_EXCEPTION("label path is empty.");
			}
			else if (data_path.empty() && !label_path.empty()) {
				THROW_EXCEPTION("data path is empty.");
			}

			xarr_f out_data, out_label;
			load_data(target, out_data);
			load_data(label, out_label, false);
			transform_data(out_data, out_label);
		}

	protected:
		void load_data(const string& path, xarr_f& out_data, const bool isData = true) {
			std::ifstream is(path, std::ifstream::binary);
			is.seekg(0, is.end);
			size_t length = static_cast<size_t>(is.tellg());
			is.seekg(0, is.beg);
			size_t offset = isData ? 16 : 8;
			int top_shape = -1;
			xarr_uc o_data;

			if (isData) {
				top_shape = (length - offset) / (28 * 28);
				o_data = xt::zeros<unsigned char>({ top_shape, 1, 28, 28 });
			}
			else {
				top_shape = length - offset;
				o_data = xt::zeros<unsigned char>({ top_shape });
			}
			is.seekg(offset);
			is.read(reinterpret_cast<char*>(o_data.data()), length - offset);
			is.close();
			out_data = xt::cast<float>(o_data);
		}

	protected:
		string data_path;
		string label_path;
	};
}

#endif