#ifndef __LOCAL_EXCEPTION_H__
#define __LOCAL_EXCEPTION_H__

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#define THROW_EXCEPTION(comment) throw LocalException(comment, __FILE__, __FUNCSIG__, __LINE__)

namespace md {
	class LocalException {
	public:
		LocalException(std::string comment) {
			err_comment += "[ERROR] ";
			err_comment += comment;
		}

		LocalException(std::string comment, const char* filename, const char* loc, int line) {
			err_comment += "[";
			err_comment += get_file_name(filename);
			err_comment += "] [";
			err_comment += loc;
			err_comment += "] / Line : ";
			err_comment += std::to_string(line);
			err_comment += " - " + comment;
		}

		std::string& get_message() {
			return err_comment;
		}
	protected:
		void print_comment() {
			std::cout << err_comment << std::endl;
		}

		std::string get_file_name(const char* filename) {
			std::stringstream ss(filename);
			std::string ret;
			while (getline(ss, ret, '\\'));
			return ret;
		}

	protected:
		std::string err_comment;
	};
}
#endif
