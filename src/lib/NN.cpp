#include "../inc/NN.hpp"
#include "../inc/mathutil.hpp"
#include "NEAT.hpp"

namespace NN {

namespace Activations {
    Float_t sigmoid(const Float_t x) {
        constexpr Float_t e = Math::Constants<Float_t>::EULER;
        return Float_t{1} / (Float_t{1} + Math::powt(e,x));
    }

    Float_t ReLU(const Float_t x) {
        return x < 0 ? 0 : x;
    }
};


BasicNetwork_s::BasicNetwork_s(const NEAT::Genome_s genome, const ActivateFunc_t activation) 
    : _connections(nullptr)
    , _node_sb(genome.getNodeCount(), true)
    , activation(activation)
    , _input_start(genome.getInputNodeStart())
    , _input_count(genome.getInputNodeCount())
    , _has_bias(genome.hasBias())
    , _bias_node(genome.getBiasNode())
    , _output_start(genome.getOutputNodeStart())
    , _output_count(genome.getOutputNodeCount())
    , _ready(false)
{
    // Verify that
    if (!_node_sb.isReady()) {
        fprintf(stderr, "ERROR: Failed to initialze Node Swap Buffer in BasicNetwork_s::BasicNetwork_s");
        return;
    }

    _connection_count = 0;
    for (const NEAT::Gene_s& gene : genome.getGenome()) {
        if (gene.enabled) ++_connection_count;
    }

    // Malloc connections
    _connections = (Connection_s*)malloc(sizeof(Connection_s)*_connection_count);
    
    if (_connections == nullptr) {
        fprintf(stderr, "ERROR: Failed to malloc for connections buffer in BasicNetwork_s::BasicNetwork_s");
        return;
    }
    
    // Use index for `_connections` and iterator for `genome.getGenome()` for consistency no matter `typeof(genome.getGenome())`
    ConnectionID_t i = 0;
    for (const NEAT::Gene_s& gene : genome.getGenome()) {
        // Skip disabled
        if (!gene.enabled) continue;
        // Copy gene
        _connections[i] = {
            .from = gene.FROM,
            .to = gene.TO,
            .weight = gene.weight,
        };
        // Increment
        ++i;
    }

    _ready = true;
}

const Float_t* BasicNetwork_s::_step(const Float_t* const inputs) {
    // Set inputs
    memcpy(
        _node_sb.getFront()+_input_start, 
        inputs, 
        _input_count
    );
    // Set bias
    if (_has_bias) _node_sb.getFront()[_bias_node].value = Float_t{1};

    // Zero back buffer
    _node_sb.zero(false, true);

    // Connection inputs
    const Node_s* inp = _node_sb.getFront();
    // Connection outputs
    Node_s* out = _node_sb.getBack();
    // Connections won't be modified, so we'll use a const pointer
    const Connection_s* connections = _connections;

    // Evaluate each connection
    for (ConnectionID_t i = 0; i < _connection_count; ++i) {
        const Connection_s& conn = connections[i];

        out[conn.to].value += inp[conn.from].value*conn.weight;
    }

    // Apply activation function
    for (NodeID_t i = 0; i < _node_sb.getCount(); ++i) {
        out[i].value = activation(out[i].value);
    }

    // Swap front and back buffers
    _node_sb.swap();
    // Outputted values are now in `_node_sb.getFront()`

    return (Float_t*)(_node_sb.getFront()+_output_start);
}

const Float_t* BasicNetwork_s::step(void* const network, const Float_t* const inputs) {
    return ((BasicNetwork_s* const)network)->_step(inputs);
}

}
