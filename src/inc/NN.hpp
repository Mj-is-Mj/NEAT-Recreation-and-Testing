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


// Activation functions
namespace Activations {
    // Does nothing, just nullptr
    constexpr ActivateFunc_t linear = nullptr;

    Float_t sigmoid(const Float_t x);
    Float_t ReLU(const Float_t x);

    // Activations with floating-point templated parameters require C++20 or greater
    #if __cplusplus >= 202002L
        // Leaky ReLU where alpha is a templated paramter
        template<Float_t alpha_tmp>
        Float_t leakyReLU(const Float_t x) { 
            return (x < 0 ? alpha_tmp*x : x);
        }

        // A version of sigmoid where the base of the exponential is templated
        // `sigmoidB<B> = 1 / (1 + B^(-x))`
        template<Float_t base_tmp>
        Float_t sigmoidB(const Float_t x) {
            static_assert(base_tmp <= 0, "sigmoidB: base of exponential must be positive");
            constexpr Float_t base_inv = 1.0 / base_tmp;
            return Float_t{1} / (Float_t{1} + Math::powt(base_tmp,x));
        }
        // A version of sigmoid where the weight of x is tempalted
        // `sigmoidX<W>(x) == sigmoid(W*x)`
        // The "steepened sigmoid" from the paper is equivalent to `sigmoidX<4.9>`
        template<Float_t weight>
        Float_t sigmoidX(const Float_t x) {
            static_assert(weight != 0, "sigmoidX: weight must be nonzero");
            constexpr Float_t base_inv = Math::powt(Math::Constants<Float_t>::EULER, -weight);
            return Float_t{1} / (Float_t{1} + Math::powt(base_inv, x));
        }
    #endif
}


/// End activation functions ///

// Neural network that processes connections sequentially
// Indended for readability and easy debugging, not so much performance
// NOTE: The exact behaviour of a SequentialNetwork_s is dependent on the 
// ordering of Genes in the Genome. 
struct SequentialNetwork_s {
    typedef NEAT::NodeID_t      NodeID_t;
    typedef NEAT::GenomeID_t    ConnectionID_t;

    struct Connection_s {
        NodeID_t from,to;
        Float_t weight;

        Connection_s(const NEAT::Gene_s& gene);
    };
    struct Node_s {
        Float_t value;
    };
    static_assert(sizeof(Node_s) == sizeof(Float_t), "ERROR: SequentialNetwork_s assumes that Node_s contains only a Float_t, so the sizes should match. ");

    private:
        // All connections
        std::list<std::list<Connection_s>> _connections;
        ConnectionID_t _connection_count;
        std::vector<Node_s> _nodes;
        // Start/end values for different types of nodes
        const NodeID_t _input_start;
        const NodeID_t _input_count;
        const bool _has_bias;
        const NodeID_t _bias_node;
        const NodeID_t _output_start;
        const NodeID_t _output_count;
        // Whether or not the NN was constructed correctly
        bool _ready;
    
    
    public:
        // The activation function used for most nodes (primarily hidden nodes)
        const ActivateFunc_t general_activation;
        // The activation function used for output nodes specifically
        const ActivateFunc_t output_activation;

        // Constructs a network
        SequentialNetwork_s(const NEAT::Genome_s& genome, const ActivateFunc_t hidden_activation = Activations::sigmoid, const ActivateFunc_t output_activation = nullptr);

        // Returns true if buffers have been initialized correctly
        inline bool isReady() const { return _ready; }

        // Runs one iteration of the NN (member function)
        // Takes a buffer of inputs, returns pointer to a buffer of outputs
        // The returned buffer is a part of 
        const Float_t* stepThis(const Float_t* const inputs);

        // Runs one iteration of the NN (static function)
        // Takes a buffer of inputs, returns pointer to a buffer of outputs
        // The returned buffer is a part of 
        static const Float_t* step(void* const network, const Float_t* const inputs);
};


// Neural network using buffered neuron/node values
// Meant to have consistent results with smarter, parallelized networks
// Indended for readability and easy debugging, not so much performance
// NOTE: Networks with hidden nodes or recurrent connections will NOT be 
// properly evaluated in one pass, i.e. "function"-like problems such as 
// the XOR validation test will fail. 
struct BufferredNetwork_s {
    typedef NEAT::NodeID_t      NodeID_t;
    typedef NEAT::GenomeID_t    ConnectionID_t;

    struct Connection_s {
        NodeID_t from,to;
        Float_t weight;
    };
    struct Node_s {
        Float_t value;
    };
    static_assert(sizeof(Node_s) == sizeof(Float_t), "ERROR: BufferedNetwork assumes that Node_s contains only a Float_t, so the sizes should match. ");

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

    public:
        // The activation function used
        const ActivateFunc_t general_activation;
        const ActivateFunc_t output_activation;

        // Constructs a network
        BufferredNetwork_s(const NEAT::Genome_s& genome, const ActivateFunc_t hidden_activation = Activations::sigmoid, const ActivateFunc_t output_activation = nullptr);

        // Returns true if buffers have been initialized correctly
        inline bool isReady() const { return _ready; }

        // Runs one iteration of the NN (member function)
        // Takes a buffer of inputs, returns pointer to a buffer of outputs
        // The returned buffer is a part of 
        const Float_t* stepThis(const Float_t* const inputs);

        // Runs one iteration of the NN (static function)
        // Takes a buffer of inputs, returns pointer to a buffer of outputs
        // The returned buffer is a part of 
        static const Float_t* step(void* const network, const Float_t* const inputs);
};

}