#include "../inc/NEAT.hpp"
#include "../inc/randutil.hpp"

// Repeat connections are not allowed to be added from "add connection"
// mutations. However, given N nodes, there are nearly N^2 possible connections,
// which would make tracking what connections are available tedious and slow.
// Since the odds of hittin repeat connections will be quite low, we instead 
// just pick random nodes and see if the connection exists. This can be reliably 
// done with very few attempts in most networks, but we set a limit just in case. 
// The one exception is the initial fully-connected networks
#ifndef MAX_TRIES_ADD_CONNECTION
    #define MAX_TRIES_ADD_CONNECTION 10
#endif

// #define WARN_MAX_TRIES_EXCEEDED


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

bool GenePool_s::addGenome(const Genome_s genome, const GenomeID_t count) {
    if (&(genome.POOL) != this) return false;

    gene_pool.reserve(gene_pool.size() + count);

    gene_pool.emplace_back(genome);

    for (GenomeID_t i = 1; i < count; ++i) {
        gene_pool.emplace_back(genome.duplicate());
    }

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

std::ostream& operator<<(std::ostream& out, const GenePool_s& pool) {
    out << "Parameters {" << std::endl
        << "\tInputs:  " << pool.INPUT_NODE_COUNT << std::endl
        << "\tOutputs: " << pool.INPUT_NODE_COUNT << std::endl
        << "\tBais:    " << (pool.HAS_BIAS_NODE ? "present" : "absent")
        << std::endl << "}" << std::endl;


    for (const auto& spec : pool.species) {
        out << "Species " << spec.ID << ": {" << std::endl
            << "\tExitinct:        " << (spec.allowed_to_reproduce ? "no" : "yes") << std::endl
            << "\tMAX Fitness OAT: " << spec.cumulative_max_fitness << std::endl
            << "\tCurrent Max Fit: " << spec.current_max_fitness<< std::endl
            << "\tCurrent Avg Fit: " << spec.current_avg_fitness << std::endl
            << "\tStaleness:       " << spec.staleness << std::endl
            << "\tMembers: {" << std::endl;
        for (const auto& gid : spec.members) {
            const auto& genome = pool.gene_pool[gid];
            out << "\t\t";
            genome.simplifiedPrint(out);
            out << ", " << std::endl;
        }
        out << "\t}" << std::endl
            << "}\n" << std::endl;
    }

    if (pool.species.size() > 0) return out;

    out << "Members: {" << std::endl;
    for (const auto& genome : pool.gene_pool) {
        out << "\t";
        genome.simplifiedPrint(out);
        out << ", " << std::endl;
    }
    out << "}" << std::endl;

    return out;
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
    , PARENT_A(ERR_VAL<NodeID_t>())
    , PARENT_B(ERR_VAL<NodeID_t>())
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
    , ID(other.ID)
    , PARENT_A(other.PARENT_A)
    , PARENT_B(other.PARENT_B)
    , node_count(other.node_count)
    , fitness(other.fitness)
    , genome(other.genome)
{}

Genome_s::Genome_s(
    const GenePool_s& pool, const GenomeID_t& ID, 
    const GenomeID_t parent_a, const GenomeID_t parent_b,
    const NodeID_t node_count, const std::vector<Gene_s>& genome
)   : POOL(pool)
    , ID(ID)
    , PARENT_A(parent_a)
    , PARENT_B(parent_b)
    , node_count(node_count)
    , fitness(0)
    , genome(genome)
{}

NodeID_t Genome_s::getRandomNodeID() const {
    return RandUtil::randUpTo(node_count);
}
NodeID_t Genome_s::getRandomHiddenNodeID() const {
    return RandUtil::randRange(getHiddenNodeStart(), getHiddenNodeEnd());
}
NodeID_t Genome_s::getRandomInputOrHiddenNodeID() const {
    const NodeID_t N = RandUtil::randUpTo(getInputNodeCount() + getHiddenNodeCount());
    if (N < getInputNodeCount()) {
        return getInputNodeStart() + N;
    }
    else {
        return getHiddenNodeStart() + N - getInputNodeCount();
    }
}
NodeID_t Genome_s::getRandomOutputOrHiddenNodeID() const {
    const NodeID_t N = RandUtil::randUpTo(getOutputNodeCount() + getHiddenNodeCount());
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

    #ifdef WARN_MAX_TRIES_EXCEEDED
        std::cerr << "WARNING: Failed to create connection: "
            << from << "->" << to << " | " 
            << __FILE__ << __LINE__ << std::endl;
    #endif

    return false;
}

bool Genome_s::mutateAddBias() { 
    const NodeID_t& BIAS = getBiasNode();
    assert(hasBias());

    NodeID_t to;
    for (size_t i = 0; i < MAX_TRIES_ADD_CONNECTION; ++i) {
        to   = getRandomOutputOrHiddenNodeID();

        if (connectionExists(BIAS, to)) continue;

        addBias(
            to, 
            getRandomWeight(),
            true
        );
        return true;
    }

    #ifdef WARN_MAX_TRIES_EXCEEDED
    std::cerr << "WARNING: Failed to create bias connection: "
        << BIAS << "->" << to << " | " 
        << __FILE__ << __LINE__ << std::endl;
    #endif

    return false;
}

bool Genome_s::mutateSetConnection(const bool enabled) {
    const GenomeID_t TARGET = getRandomConnectionID();
    
    // Try every connection, starting with the randomly selected one
    GenomeID_t i = TARGET;
    do {
        Gene_s& gene = genome[i];
        if (gene.enabled != enabled) {
            gene.enabled = enabled;
            return true;
        }

        i = (i+1) % getGenomeSize();
    } while (i != TARGET);

    #ifdef WARN_MAX_TRIES_EXCEEDED
        std::cerr << "WARNING: Failed to " << (enabled ? "enable" : "disable") 
            << " a connection." << __FILE__ << __LINE__ << std::endl;
    #endif
    return false;
}

void Genome_s::mutateSetRandomWeight() {
    getRandomGenome().weight = getRandomWeight();
}

void Genome_s::mutatePerturbWeight() {
    getRandomGenome().weight += getRandomWeightPerturbation();
}

void Genome_s::mutate() {
    #define REPEAT(N) for (\
        size_t i = 0; \
        i < RandUtil::randCount<size_t>(N);\
        ++i \
    )

    const auto& PARAMS = POOL.PARAMETERS.reproduction.mutation;

    REPEAT(PARAMS.rates.add_node) {
        mutateAddNode();
    }

    REPEAT(PARAMS.rates.add_connection) {
        mutateAddConnection();
    }

    REPEAT(PARAMS.rates.add_bias) {
        mutateAddBias();
    }
    
    REPEAT(PARAMS.rates.disable_connection) {
        mutateDisableConnection();
    }
    
    REPEAT(PARAMS.rates.enable_connection) {
        mutateEnableConnection();
    }
    
    REPEAT(PARAMS.rates.weight) {
        if (RandUtil::randProb(PARAMS.weight_perturb_chance)) {
            mutatePerturbWeight();
        }
        else {
            mutateSetRandomWeight();
        }
    }

    #undef REPEAT
}


Genome_s Genome_s::makeMutatedClone(const Genome_s &parent) {
    Genome_s clone = parent.clone();
    clone.mutate();
    return clone;
}


// There was a misinterpretation I had: I had thought that 
// Matching, Disjoint, and Excess genes formed contiguous blocks,
// when you can actually have Disjoint genes sitting between Matching
// genes
Genome_s Genome_s::crossover(const Genome_s &pA, const Genome_s &pB) {
    #define INNOV_NUM(G,i) (G.genome[i].INNOVATION_NUM)

    assert(&(pA.POOL) == &(pB.POOL));
    const GenePool_s& POOL = pA.POOL;

    // Generate child with no genome
    Genome_s child = Genome_s(
        POOL, POOL.getNextGenomeNumber(), 
        pA.ID, pB.ID,
        std::max(pA.node_count, pB.node_count),
         {}
    );


    GenomeID_t iA=0, iB=0;
    bool A_has_excess;

    while (true) {
        if (iA >= pA.getGenomeSize()) {
            A_has_excess = false;
            break;
        }
        if (iB >= pB.getGenomeSize()) {
            A_has_excess = true;
            break;
        }

        const Gene_s& gA = pA.genome[iA];
        const Gene_s& gB = pB.genome[iB];

        // If genes are disjoint, add genes in order of innovation number
        if (gA.INNOVATION_NUM != gB.INNOVATION_NUM) {
            if (gA.INNOVATION_NUM < gB.INNOVATION_NUM) {
                child.genome.push_back(gA);
                ++iA;
            }
            else {
                child.genome.push_back(gB);
                ++iB;
            }
            continue;
        }
        // Otherwise, genes are matching

        // Select a gene from one parent randomly
        child.genome.push_back(RandUtil::randCoinFlip() ? gA : gB);

        // "There was a 75% chance that an inherited gene was disabled if 
        // it was disabled in either parent." 
        // I'm not sure if "was disabled" means "the result was a disabled gene"
        // or "was disabled through an added chance to disable". I'm also not 
        // sure if "either" is exclusive or inclusive, 
        // I interpretted this as the probability of the result and an exclusive 
        // "either" respectively. 
        if (gA.enabled ^ gB.enabled) {
            child.genome.back().enabled = ! RandUtil::randProb(
                POOL.PARAMETERS.reproduction.crossover.keep_disabled_connection
            );
        }

        // Increment both indicies
        ++iA; ++iB;
    }

    // Select parent that has excess
    const Genome_s& pE = A_has_excess ? pA : pB;
    NodeID_t& iE = A_has_excess ? iA : iB;

    // Copy excess
    for (/*iE*/; iE < pE.getGenomeSize(); ++iE) {
        child.genome.push_back(pE.genome[iE]);
    }

    return child;

    #undef INNOV_NUM
}

Float_t Genome_s::compatibilityDistance(const Genome_s& A, const Genome_s& B) {
    #define INNOV_NUM(G,i) (G.genome[i].INNOVATION_NUM)
    assert(&(A.POOL) == &(B.POOL));

    GenomeID_t disjoint=0, matching=0;
    Float_t weight_diff = 0;

    GenomeID_t iA=0, iB=0;
    while (iA < A.getGenomeSize() && iB < B.getGenomeSize()) {

        // If disjoint
        if (INNOV_NUM(A,iA) != INNOV_NUM(B,iB)) {
            // Increment index of lowest innovation number
            if (INNOV_NUM(A,iA) < INNOV_NUM(B,iB))
                ++iA;
            else
                ++iB;

            ++disjoint;
            continue;
        }
        // Otherwise matching

        // Add weight difference
        weight_diff += std::abs(
            (Float_t)(A.genome[iA].weight)
            - (Float_t)(B.genome[iB].weight)
        );

        // Increment both indicies
        ++iA; ++iB;

        ++matching;
    }

    #undef INNOV_NUM
    
    /* 
    For arbitrary genomes C,D:
    `C.size == C.matching(D) + C.disjoint(D) + C.excess(D)`  
    So `C.excess(D) + D.excess(C) ==  
          C.size + D.size 
        - C.matching(D) + D.matching(C) 
        - C.disjoint(D) + D.disjoint(C)`  
    Since `matching == C.matching(D) == D.matching(C)` 
    and `disjoint == C.disjoint(D) + D.disjoint(C)`, the
    formula below holds. 
    */
    const GenomeID_t excess = A.getGenomeSize() + B.getGenomeSize() - (2*matching + disjoint);
    const GenomeID_t N = std::max(A.getGenomeSize(), B.getGenomeSize());

    if (matching > 0)
        weight_diff /= matching;
    else
        weight_diff = 0;

    const auto& CDF = A.POOL.PARAMETERS.cdf;    

    return ((CDF.c1*excess + CDF.c2*disjoint) / N) + CDF.c3*weight_diff;
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

void Genome_s::simplifiedPrint(std::ostream& out) const {
    out << "Genome " << ID << ": (";
    if (!IS_ERR(PARENT_A)) out << PARENT_A;
    if (!IS_ERR(PARENT_B)) out << "x" << PARENT_B;
    out << ") {";
    for (const auto& gene : genome) {
        out << gene.INNOVATION_NUM << ", ";
    }
    out << "}";
}


}