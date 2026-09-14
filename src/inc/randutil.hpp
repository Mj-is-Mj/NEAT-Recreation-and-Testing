#pragma once

#include <cstdlib>
#include <cassert>

namespace RandUtil {

#if RAND_MAX < 0x3FFFFFFF
// version of rand() that guarentees 30 bits instead of 15 by xor'ing two calls to `rand()`
#define rand30() (rand() ^ (rand() << 15))
#else
// version of rand() that guarentees 30 bits instead of 15, but your system gives 30+ bits already so this is just `rand()`
#define rand30() (rand())
#endif


// Returns a random floating point value [0.0, 1.0]
template<typename float_t = float>
inline float_t randF() {
    return (float_t)(rand()) / (float_t)RAND_MAX;
}

// Returns a random number in the range (0,n]
template<typename num_t>
inline num_t randUpTo(const num_t n) {
    return (num_t)(rand30()) % n;
}

// Returns a random number in the range (s,e]
template<typename num_t>
inline num_t randRange(const num_t s, const num_t e) {
    return s + randUpTo(e-s);
}

// Returns true or false using `prob` as the probability of true
template<typename float_t>
inline bool randProb(const float_t prob) {
    return randF<float_t>() < prob;
}

// Equivalent to randProb(0.5)
inline bool randCoinFlip() {
    return rand() & 0b1;
}

// Generates a partially random whole number. 
  // Returns (whole number portion of avg_count + fractional portion treated as the probability of being 1 rather than 0)
  // E.g. 4.3 has a 70% chance of return 4 and a 30% chance of returning 5, returning 4.3 on average
template<typename count_t = size_t, typename float_t>
inline count_t randCount(const float_t avg_count) {
    count_t count = (count_t)avg_count;

    float_t remainder = avg_count - (float_t)count;

    if ( remainder > 0 && randProb(remainder) ) {
        return count + 1;
    }

    return count;
}

// Generates a pair of unique integers in [0,`range`)
template<typename count_t>
inline void randUniquePair(count_t& a, count_t& b, const count_t range) {
    a = (count_t)rand30() % range;
    b = (count_t)rand30() % (range-1);

    if (b >= a) ++b;
}

// Generates a random pair of integers with distinct but overlapping ranges
  // E.g. randUniquePair(x,y,0,20,10,30) will generate a pair (x,y) 
  // s.t. x in [0,20), y in [10,30), x!=y
template<typename count_t>
inline void randUniquePair(count_t& a, count_t& b, const count_t s1, const count_t e1, const count_t s2, const count_t e2) {
    const count_t d1 = e1-s1;
    const count_t d2 = e2-s2;
    assert(d1 > 0 && d2 > 0);

    a = s1 + ((count_t)rand30() % d1);

    if (a < e2 && a >= s2)
        b = s2 + ((count_t)rand30() % (d2-1));
    else
        b = s2 + ((count_t)rand30() % d2);
    
    if (b >= a) b++;
}

}