#pragma once

#include <cstdlib> // For size_t, malloc, calloc, realloc
#include <cstring> // For memset

struct RawSwapBuffer_s {
    private:
        size_t _size;
        void* _front;
        void* _back;
        bool _ready;

        // Initializes buffers to size of `this->size`
        int _init(const bool zero = true);
        // Resizes buffers using the current value of `this->size`
        int _resize(const bool zero = true);

    public:
        // Do nothing
        inline RawSwapBuffer_s() : _size(0), _front(nullptr), _back(nullptr), _ready(false) {};
        // Creates buffers with the specified size, sets contents to zero when `zero=true`
        RawSwapBuffer_s(const size_t size, const bool zero = true);

        // Resize the front and back buffers to the new size. `zero=true` ==> 
        int resize(const size_t new_size, const bool zero = true);
        // Free all buffers and leave nullptr behind
        void clean();

        // Set value of buffers to a particular pattern. 
        // Returns `buffer_size % psize`. If nonzero, then that number of bytes at the end of each buffer was not set. 
        int setPattern(const void* const pattern, const size_t psize, const bool front = true, const bool back = true);
        // Zeros out one or both buffers
        void zero(const bool front = true, const bool back = true);

        // Returns true if both buffers have been initialized correctly
        inline bool isReady() const { return _ready; }
        // Get size of buffers
        inline size_t getSize() const { return _size; }

        // Swaps front and back buffers
        inline void swap() { void* tmp = _front; _front = _back; _back = tmp; }
        // Returns the current front-buffer
        inline void* getFront() { return _front; }
        // Returns the current back-buffer
        inline void* getBack() { return _back; }
};

// Explicitly-typed wrapper for `RawSwapbuffer_s` to avoid frequen pointer conversions and whatnot
template<typename Data_tmp>
struct SwapBuffer_s {
    private:
        // The number of elements in each buffer
        size_t _count;
        // The actual buffers with swapping functionality
        RawSwapBuffer_s _sb;
    
    public:
        // Do nothing
        inline SwapBuffer_s<Data_tmp>() : _sb(), _count(0) {}
        // Creates buffers with the specified size, sets contents to zero when `zero=true`
        SwapBuffer_s<Data_tmp>(const size_t count, const bool zero = true) : _count(count), _sb(_count*sizeof(Data_tmp), zero) {}


        // Returns true if both buffers have been initialized correctly
        inline bool isReady() const { return _sb.isReady(); };
        // Get size of each buffer in bytes
        inline size_t getSize() const { return _sb.getSize(); }
        // Get number of elements in each buffer
        inline size_t getCount() const { return _count; }

        // Swaps front and back buffers
        inline void swap() { _sb.swap(); }
        // Returns the current front-buffer
        inline Data_tmp* getFront() { return (Data_tmp*)_sb.getFront(); }
        // Returns the current back-buffer
        inline Data_tmp* getBack()  { return (Data_tmp*)_sb.getBack(); }
        
        // Copies the given value to each element of each buffer
        // Returns `buffer_size % psize`. If nonzero, then that number of bytes at the end of each buffer was not set. 
        inline int setValue(const Data_tmp value, const bool front=true, const bool back=true) {
            return _sb.setPattern((void*)&value, sizeof(Data_tmp), front, back);
        }
        // Zeros out one or both buffers
        inline void zero(const bool front = true, const bool back = true) { _sb.zero(front, back); }
};