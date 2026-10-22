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

// Power function
template<typename f_tmp>
constexpr f_tmp pow(const f_tmp b, const f_tmp e) { return std::pow(b, e); }

// Basic trig functions
template<typename f_tmp>
constexpr f_tmp cos(const f_tmp theta) { return std::cos(theta); }
template<typename f_tmp>
constexpr f_tmp sin(const f_tmp theta) { return std::sin(theta); }
template<typename f_tmp>
constexpr f_tmp tan(const f_tmp theta) { return std::tan(theta); }

template<typename f_tmp>
constexpr f_tmp cos2(const f_tmp theta) { f_tmp r = Math::cos(theta); return r*r; }
template<typename f_tmp>
constexpr f_tmp sin2(const f_tmp theta) { f_tmp r = Math::sin(theta); return r*r; }

template<typename f_tmp>
constexpr f_tmp sgn(const f_tmp x) { return (x < 0 ? f_tmp{-1} : (x > 0) ? f_tmp{1} : 0); }

}