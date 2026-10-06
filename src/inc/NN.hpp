#pragma once
#include "../inc/NEAT.hpp"
#include "../inc/Swapbuffer.hpp"
#include "../inc/mathutil.hpp"

namespace NN {

    // TYPEDEFS: Used to avoid ambiguity of different number types
    // Note: types with the same name will be used in other namespaces
    // You should not use `using namespace X;` for multiple namespaced defined
    // in this project, or anywhere inside of a namespace. 

// Floating point type used for all NN computation
typedef float   Float_t;
// Activation function type used for all activations
typedef Float_t (*ActivateFunc_t)(const Float_t);
// Step function used for all networks
typedef const Float_t* (*StepFunc_t)(void* const network, const Float_t* const inputs);


namespace Activations {
    Float_t sigmoid(const Float_t x);
    Float_t ReLU(const Float_t x);

    // Leaky ReLU where \alpha equals `(Float_t)num_tmp/(Float_t)div_tmp`
    template<size_t numerator_tmp, size_t divisor_tmp>
    Float_t leakyReLU(const Float_t x) { 
        static_assert(divisor_tmp != 0, "leakyReLU: Alpha divisor must not be zero");
        constexpr Float_t alpha = (Float_t)numerator_tmp / (Float_t)divisor_tmp;
        return (x < 0 ? alpha*x : x);
    }

    // A version of sigmoid where the base of the exponential is repersented using
    // a numerator and divisor
    template<size_t numerator_tmp, size_t divisor_tmp>
    Float_t sigmoidB(const Float_t x) {
        static_assert(divisor_tmp != 0, "sigmoidB: Base divisor must not be zero");
        constexpr Float_t base = (Float_t)numerator_tmp / (Float_t)divisor_tmp;
        constexpr Float_t e = Math::Constants<Float_t>::EULER;
        return Float_t{1} / (Float_t{1} + Math::powt(e,x));
    }
}

/// End activation functions ///


// A basic implementation of a neural network
// Indended for readability and easy debugging
// Meant to have consistent results with smarter, parallelized networks
struct BasicNetwork_s {
    struct Connection_s;
    struct Node_s;

    typedef NEAT::NodeID_t      NodeID_t;
    typedef NEAT::GenomeID_t    ConnectionID_t;

    struct Connection_s {
        NodeID_t from,to;
        Float_t weight;
    };
    struct Node_s {
        Float_t value;
    };
    static_assert(sizeof(Node_s) == sizeof(Float_t), "ERROR: BasicNetwork assumes that Node_s contains only a Float_t, so the sizes should match. ");

    private:
        // All connections
        Connection_s* _connections;
        ConnectionID_t _connection_count;
        // Buffers for node values
        SwapBuffer_s<Node_s> _node_sb;
        // Start/end values for different types of nodes
        const NodeID_t _input_start;
        const NodeID_t _input_count;
        const bool _has_bias;
        const NodeID_t _bias_node;
        const NodeID_t _output_start;
        const NodeID_t _output_count;
        // Whether or not the NN was constructed correctly
        bool _ready;

        // Runs one iteration of the NN
        // Takes a buffer of inputs, returns pointer to a buffer of outputs
        // The returned buffer is a part of 
        const Float_t* _step(const Float_t* const inputs);

    public:
        // The activation function used
        const ActivateFunc_t activation;

        // Constructs a network
        BasicNetwork_s(const NEAT::Genome_s genome, const ActivateFunc_t activation = Activations::sigmoid);

        // Returns true if buffers have been initialized correctly
        inline bool isReady() const { return _ready; }

        static const Float_t* step(void* const network, const Float_t* const inputs);
};

}