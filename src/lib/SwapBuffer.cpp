#include "../inc/Swapbuffer.hpp"



RawSwapBuffer_s::RawSwapBuffer_s(const size_t size, const bool zero) 
    : _size(size)
    , _front(nullptr)
    , _back(nullptr)
    , _ready(false)
{ _init(zero); }

int RawSwapBuffer_s::_init(const bool zero) {
    // Allocate buffers
    if (zero) {
        _front = calloc(_size, 1);
        _back  = calloc(_size, 1);
    }
    else {
        _front = malloc(_size);
        _back  = malloc(_size);
    }

    // Check success
    int ret = (_front ? 0 : 1) + (_back ? 0 : 2);
    _ready = (bool)(ret == 0);

    return ret;
}

int RawSwapBuffer_s::_resize(const bool zero) {
    // realloc or malloc front
    if (_front)
        _front = realloc(_front, _size);
    else
        _front = malloc(_size);

    // realloc or malloc front
    if (_back)
        _back = realloc(_back, _size);
    else
        _back = malloc(_size);

    // Set values to 0
    if (_back && zero) memset(_back, 0, _size);
    if (_front && zero) memset(_front, 0, _size);

    // Check success
    int ret = (_front ? 0 : 1) + (_back ? 0 : 2);
    _ready = (bool)(ret == 0);

    return ret;
};

int RawSwapBuffer_s::resize(const size_t new_size, const bool zero) {
    this->_size = new_size;
    this->_ready = false;
    return _resize(zero);
};

void RawSwapBuffer_s::clean() {
    _ready = false;
    if (_front) {
        free(_front);
        _front = nullptr;
    }
    if (_back) {
        free(_back);
        _back = nullptr;
    }
}

// Similar to `memset`, but can copy arbitrarily sized patterns rather than single bytes
// A section of length `dsize % psize` at the end of `dest` will be left unmodified, i.e.
// the entirety of `dest` will only be modified if `dsize % psize == 0`
void memset_pattern(void* const dest, const void* const pattern, const size_t psize, const size_t dsize) {
    // If pattern is a single byte, just do regular memset
    if (psize == 1) {
        memset(dest, (int)(*(unsigned char*)pattern), dsize);
        return;
    }
    
    unsigned char* ptr = (unsigned char*)dest;
    unsigned char* const end = ptr + psize*(dsize - dsize%psize);

    while (ptr < end) {
        memcpy(ptr, pattern, psize);
        ptr += psize;
    };
};

int RawSwapBuffer_s::setPattern(const void* const pattern, const size_t psize, const bool front, const bool back) {

    if (back) memset_pattern(this->_back, pattern, psize, this->_size);
    if (front) memset_pattern(this->_front, pattern, psize, this->_size);

    return (this->_size % psize);
};

void RawSwapBuffer_s::zero(const bool front, const bool back) {
    if (_front && front) memset(_front, 0, _size);
    if (_back && back) memset(_back, 0, _size);
}