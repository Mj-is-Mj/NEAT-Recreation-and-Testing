#pragma once

#include <cmath>

namespace Math {

// Constants (Double)
template<typename Type>
struct Constants {
    #define MAKE_CONST(name, literal) static constexpr Type name = static_cast<Type>(literal);
    
    MAKE_CONST(PI,      3.1415926535)
    MAKE_CONST(EULER,   2.7182818284)
};

// A templated version of pow that selects the correct version to use
template<typename f_tmp>
inline f_tmp powt(const f_tmp b, const f_tmp e);

template<> inline float powt<float>(const float b, const float e) { return powf(b,e); };
template<> inline double powt<double>(const double b, const double e) { return pow(b,e); };



}