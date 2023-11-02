#pragma once

#if defined(U_OS_WINDOWS)
#	define HAS_UUID
#	include <Windows.h>
#   include <wingdi.h>
#	include <Shlwapi.h>
#	include <winsock2.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#endif


std::vector<std::string> split_string(const string& str_datas, string seg_sign = ",") {
	std::string str = str_datas;
	std::vector<std::string> vec;
	std::stringstream ss(str);
	std::string token;
	while (getline(ss, token, ',')) {
		vec.push_back(token);
	}
	for (auto&& s : vec) {
		std::cout << s << std::endl;
	}

	return vec;
}


string i_format(const char* fmt, ...) {
	va_list vl;
	va_start(vl, fmt);
	char buffer[2048];
	vsnprintf(buffer, sizeof(buffer), fmt, vl);
	return buffer;
}



std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
	const int h_i = static_cast<int>(h * 6);
	const float f = h * 6 - h_i;
	const float p = v * (1 - s);
	const float q = v * (1 - f * s);
	const float t = v * (1 - (1 - f) * s);
	float r, g, b;
	switch (h_i) {
	case 0:r = v; g = t; b = p; break;
	case 1:r = q; g = v; b = p; break;
	case 2:r = p; g = v; b = t; break;
	case 3:r = p; g = q; b = v; break;
	case 4:r = t; g = p; b = v; break;
	case 5:r = v; g = p; b = q; break;
	default:r = 1; g = 1; b = 1; break;
	}
	return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
	float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
	float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
	return hsv2bgr(h_plane, s_plane, 1);
}
