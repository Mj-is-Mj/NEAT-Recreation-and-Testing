#include "../inc/NEAT.hpp"
#include "../inc/randutil.hpp"

// Repeat connections are not allowed to be added from "add connection"
// mutations. However, given N nodes, there are nearly N^2 possible connections,
// which would make tracking what connections are available tedious and slow.
// Since the odds of hittin repeat connections will be quite low, we instead 
// just pick random nodes and see if the connection exists. This can be reliably 
// done with very few attempts in most networks, but we set a limit just in case. 
#ifndef MAX_TRIES_ADD_CONNECTION
    #define MAX_TRIES_ADD_CONNECTION 10
#endif


namespace NEAT {

/// GENE POOL ///

void GenePool_s::clear() {
    has_been_evaluated = false;
    staleness = false;
    innovation_num = 0;
    genome_num = 0;
    species_num = 0;
    generation_num = 0;
    gene_pool.clear();
    species.clear();
}

bool GenePool_s::addGenome(const Genome_s &genome) {
    if (&(genome.POOL) != this) return false;

    gene_pool.push_back(genome);
    return true;
}


NodeID_t GenePool_s::getRandomInputNodeID() const {
    return RandUtil::randRange(
        getInputNodeStart(),
        getInputNodeEnd()
    );
}

NodeID_t GenePool_s::getRandomOutputNodeID() const {
    return RandUtil::randRange(
        getOutputNodeStart(),
        getOutputNodeEnd()
    );
}

NodeType_e GenePool_s::getNodeType(const NodeID_t n) const {
    if (isInputNode(n)) {
        return NodeType_e::INPUT;   
    }
    else if (isBiasNode(n)) {
        return NodeType_e::BIAS;   
    }
    else if (isOutputNode(n)) {
        return NodeType_e::OUTPUT;   
    }
    else if (IS_ERR(n)) {
        return NodeType_e::ERROR;   
    }
    else {
        return NodeType_e::HIDDEN;
    }
}

void GenePool_s::printNode(std::ostream& out, const NodeID_t n) const {
    switch (getNodeType(n)) {
        case NodeType_e::INPUT:
            out << n << "i";
            return;
        case NodeType_e::BIAS:
            out << n << "b";
            return;
        case NodeType_e::OUTPUT:
            out << n << "o";
            return;
        case NodeType_e::HIDDEN:
            out << n << "h";
            return;
        case NodeType_e::ERROR:
            out << "ERR";
            return;
    }
}


/// GENE ///
std::ostream& operator<<(std::ostream& out, const Gene_s& gene) {
    out << gene.INNOVATION_NUM << ": {";
    out << gene.FROM << "->" << gene.TO << ", ";
    out.precision(3);
    out << gene.weight << ", ";
    out << (gene.enabled ? "enab" : "disb") << "}";
    return out;
}

/// GENOME ///
Genome_s::Genome_s(const GenePool_s& pool, const bool fully_connect) 
    : POOL(pool)
    , ID(pool.getNextGenomeNumber())
    , node_count(pool.INPUT_NODE_COUNT + pool.OUTPUT_NODE_COUNT)\
    , fitness(0)
    , genome()
{
    for (NodeID_t i = getInputNodeStart(); i < getInputNodeEnd(); ++i) {
    for (NodeID_t j = getOutputNodeStart(); j < getOutputNodeEnd(); ++j) {
        addConnection(i, j);
    }}

    if (!hasBias()) return;

    for (NodeID_t j = getOutputNodeStart(); j < getOutputNodeEnd(); ++j) {
        addBias(j);
    }
}

Genome_s::Genome_s(const Genome_s& other) 
    : POOL(other.POOL)
    , ID(POOL.getNextGenomeNumber())
    , node_count(POOL.INPUT_NODE_COUNT + POOL.OUTPUT_NODE_COUNT)\
    , fitness(0)
    , genome(other.genome)
{}

NodeID_t Genome_s::getRandomNodeID() const {
    return RandUtil::randUpTo(node_count);
}
NodeID_t Genome_s::getRandomHiddenNodeID() const {
    return RandUtil::randRange(getHiddenNodeStart(), getHiddenNodeEnd());
}
NodeID_t Genome_s::getRandomInputOrHiddenNodeID() const {
    const NodeID_t N = RandUtil::randUpTo(getRandomInputNodeID());
    if (N < getInputNodeCount()) {
        return getInputNodeStart() + N;
    }
    else {
        return getHiddenNodeStart() + N - getInputNodeCount();
    }
}
NodeID_t Genome_s::getRandomOutputOrHiddenNodeID() const {
    const NodeID_t N = RandUtil::randUpTo(getRandomOutputNodeID());
    if (N < getOutputNodeCount()) {
        return getOutputNodeStart() + N;
    }
    else {
        return getHiddenNodeStart() + N - getOutputNodeCount();
    }
}
GeneID_t Genome_s::getRandomGeneID() const {
    return RandUtil::randUpTo(getGenomeSize());
}

bool Genome_s::connectionExists(const NodeID_t from, const NodeID_t to) {
    for (const Gene_s& gene : genome) {
        if (from == gene.FROM && to == gene.TO) {
            return true;
        }
    }
    return false;
}

void Genome_s::addConnection(
    const NodeID_t from, const NodeID_t to, 
    const Float_t weight, const bool enabled
) {
    genome.emplace_back(
        POOL.getNextInnovationNumber(),
        from, to,
        weight, enabled
    );
}

void Genome_s::addNode(Gene_s& connection) {
    // the old connection is disabled
    connection.enabled = false;
    // The connection leading into the new node gets a weight of 1
    addConnection(connection.FROM, node_count, 1, true);
    // The connection leading from the new node gets the old connection's weight
    addConnection(node_count, connection.TO, connection.weight, true);
    // ** Not specified in the paper: Does the above connection recieve `connection.enabled`,
    // or is the connection set to be enabled or disabled upon craetion?

    ++node_count;
}

void Genome_s::mutateAddNode() {
    addNode(genome[getRandomGeneID()]);
}

bool Genome_s::mutateAddConnection() {
    const Float_t& WEIGHT_RANGE = POOL.PARAMETERS.reproduction.mutation.weight_random_range;

    NodeID_t from, to;
    for (size_t i = 0; i < MAX_TRIES_ADD_CONNECTION; ++i) {
        from = getRandomInputOrHiddenNodeID();
        to   = getRandomOutputOrHiddenNodeID();

        if (connectionExists(from, to)) continue;

        addConnection(
            from, to, 
            (RandUtil::randF()-0.5) * (WEIGHT_RANGE*2),
            true
        );
        return true;
    }

    std::cerr << "WARNING: Failed to create connection: "
        << from << "->" << to << " | " 
        << __FILE__ << __LINE__ << std::endl;

    return false;
}

bool Genome_s::mutateAddBias() { 
    const Float_t& WEIGHT_RANGE = POOL.PARAMETERS.reproduction.mutation.weight_random_range;
    const NodeID_t& BIAS = getBiasNode();
    assert(hasBias());

    NodeID_t to;
    for (size_t i = 0; i < MAX_TRIES_ADD_CONNECTION; ++i) {
        to   = getRandomOutputOrHiddenNodeID();

        if (connectionExists(BIAS, to)) continue;

        addBias(
            to, 
            (RandUtil::randF()-0.5) * (WEIGHT_RANGE*2),
            true
        );
        return true;
    }

    std::cerr << "WARNING: Failed to create bias connection: "
        << BIAS << "->" << to << " | " 
        << __FILE__ << __LINE__ << std::endl;

    return false;
}



std::ostream& operator<<(std::ostream& out, const Genome_s& genome) {
    out << "Genome " << genome.ID << ": {" << std::endl
        << "\tNode count = " << genome.node_count << ", " << std::endl
        << "\tFitness = " << genome.fitness << ", " << std::endl
        << "\tGenes = {" << std::endl;

    out.precision(3);
    for (const auto& gene : genome.genome) {
        out << "\t\t" << gene.INNOVATION_NUM << ": {(";
        genome.printNode(out, gene.FROM);
        out << "->";
        genome.printNode(out, gene.TO);
        out << "), " << gene.weight << ", "
            << (gene.enabled ? "enab" : "disb") << "}\n";
    }

    out << "\t}" << std::endl << "}" << std::endl;
    return out;
}


}