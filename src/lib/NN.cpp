#include "../inc/NN.hpp"
#include "../inc/mathutil.hpp"

namespace NN {

namespace Activations {
    Float_t sigmoid(const Float_t x) {
        constexpr Float_t e = Math::Constants<Float_t>::EULER;
        return Float_t{1} / (Float_t{1} + Math::powt(e,-x));
    }

    Float_t ReLU(const Float_t x) {
        return x < 0 ? 0 : x;
    }
};


SequentialNetwork_s::Connection_s::Connection_s(const NEAT::Gene_s& gene)
    : from(gene.FROM)
    , to(gene.TO)
    , weight(gene.weight)
{}

SequentialNetwork_s::SequentialNetwork_s(const NEAT::Genome_s& genome, const ActivateFunc_t hidden_activation, const ActivateFunc_t output_activation) 
    : _connections()
    , _connection_count(0)
    , _nodes({})
    , general_activation(hidden_activation)
    , output_activation(output_activation)
    , _input_start(genome.getInputNodeStart())
    , _input_count(genome.getInputNodeCount())
    , _has_bias(genome.hasBias())
    , _bias_node(genome.getBiasNode())
    , _output_start(genome.getOutputNodeStart())
    , _output_count(genome.getOutputNodeCount())
    , _ready(false)
{
    // Initialize nodes
    _nodes.reserve(genome.getNodeCount());
    for (size_t i = 0; i < genome.getNodeCount(); ++i) {
        _nodes.push_back(Node_s{.value=0});
    }

    // Get enabled connection count
    for (const NEAT::Gene_s& gene : genome.getGenome()) {
        if (gene.enabled) ++_connection_count;
    }

    // Get list version of genome
    std::list<NEAT::Gene_s> genome_list(genome.getGenome().begin(), genome.getGenome().end());

    // Add all connections, grouped by their "to" node
    while (!genome_list.empty()) {
        // Get iterator
        auto itr = genome_list.begin();

        // Add first genome to list of connections, then erase from genome list
        _connections.push_front({Connection_s(*itr)});
        itr = genome_list.erase(itr);

        // Get new connection list and corresponding "to" node
        auto& latest_connections = _connections.front();
        const ConnectionID_t current_to_node = latest_connections.front().to;

        // Add all connections with the same "to" node to this list
        while (itr != genome_list.end()) {
            if (itr->TO == current_to_node) {
                latest_connections.push_back(*itr);
                itr = genome_list.erase(itr);
            }
            else {
                ++itr;
            }
        }
    }

    _ready = true;
}

const Float_t* SequentialNetwork_s::stepThis(const Float_t* const inputs) {
    // Copy input values into nodes
    for (size_t i = 0; i < _input_count; ++i) {
        _nodes[i+_input_start].value = inputs[i];
    }

    for (const auto& conn_list : _connections) {
        const ConnectionID_t TO = conn_list.back().to;
        Float_t out_val = 0;

        // Evaluate each connection for the given to/destination
        for (const auto& conn : conn_list) {
            out_val += conn.weight * _nodes[conn.from].value;
        }

        // Perform activation corresponding to node type
        if (_output_start <= TO && TO < _output_start+_output_count)
            _nodes[TO].value = general_activation(out_val);
        else
            _nodes[TO].value = output_activation(out_val);
    }

    return (Float_t*)(_nodes.data()+_output_start);
}

const Float_t* SequentialNetwork_s::step(void* const network, const Float_t* const inputs) {
    return ((SequentialNetwork_s* const)network)->stepThis(inputs);
}




BufferredNetwork_s::BufferredNetwork_s(const NEAT::Genome_s& genome, const ActivateFunc_t hidden_activation, const ActivateFunc_t output_activation) 
    : _connections(nullptr)
    , _connection_count(0)
    , _node_sb(genome.getNodeCount(), true)
    , general_activation(hidden_activation)
    , output_activation(output_activation)
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

const Float_t* BufferredNetwork_s::stepThis(const Float_t* const inputs) {
    // Maybe I'm tired but memcpy is not copying `inputs` -> `_node_sb.getFront()` for some reason
    // I'm doing something horribly wrong
    // Set inputs
    // memcpy(
    //     _node_sb.getFront()+_input_start, 
    //     inputs, 
    //     _input_count
    // );
    for (size_t i = _input_start; i < _input_start+_input_count; ++i) {
        _node_sb.getFront()[i].value = inputs[i];
    }

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
        Node_s& no = out[conn.to];
        const Node_s& ni = inp[conn.from];

        no.value += ni.value*conn.weight;
        no.value += 0;
    }
    
    // Apply activation for outputs
    if (output_activation)
        for (NodeID_t i = _output_start; i < _output_start+_output_count; ++i) {
            Node_s& n = out[i];
            n.value = output_activation(n.value);
        }

    // Apply activation function for all remaining node
    if (general_activation) {
        for (NodeID_t i = 0; i < _output_start; ++i)
            out[i].value = general_activation(out[i].value);
        for (NodeID_t i = _output_start+_output_count; i < _node_sb.getCount(); ++i)
            out[i].value = general_activation(out[i].value);
    }
    

    // Swap front and back buffers
    _node_sb.swap();
    // Outputted values are now in `_node_sb.getFront()`

    return (Float_t*)(_node_sb.getFront()+_output_start);
}

const Float_t* BufferredNetwork_s::step(void* const network, const Float_t* const inputs) {
    return ((BufferredNetwork_s* const)network)->stepThis(inputs);
}

}
