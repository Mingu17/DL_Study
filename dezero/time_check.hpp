#ifndef __TIME_CHECK_H__
#define __TIME_CHECK_H__

#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace chrono;
using std::string;

namespace md {
	typedef system_clock::time_point time_c;

	class TimeCheck {
		static void check_start(time_c& start) {
			start = system_clock::now();
		}

		static void check_end_nano(time_c& start, const string& header = "") {
			nanoseconds nanos = duration_cast<nanoseconds>(system_clock::now() - start);
			std::cout << header << " (nanoseconds) : " << nanos.count() << " us" << std::endl;
		}

		static void check_end_millis(time_c& start, const string& header = "") {
			milliseconds millis = duration_cast<milliseconds>(system_clock::now() - start);
			std::cout << header << " (milliseconds) : " << millis.count() << " ms" << std::endl;
		}
	};
}
#endif
