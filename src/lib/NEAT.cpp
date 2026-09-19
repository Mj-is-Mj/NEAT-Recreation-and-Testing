#include "../inc/NEAT.hpp"
#include "../inc/randutil.hpp"
#include <cstdlib>
#include <vector>

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

void GenePool_s::updateSpeciesFitnessStats() {
    for (Species_s& spec : species) {
        // Skip extinct species
        if (spec.extinct || spec.getPopSize() <= 0) {
            continue;
        }

        spec.current_max_fitness = 0;
        spec.current_avg_fitness = 0;
        spec.allotted_offspring = ERR_VAL<GenomeID_t>();

        for (const GenomeID_t gid : spec.members) {
            const Genome_s& genome = gene_pool[gid];

            // Update max fitness values
            if (genome.fitness > spec.current_max_fitness) {
                spec.current_max_fitness = genome.fitness;
                if (genome.fitness > spec.cumulative_max_fitness) {
                    spec.cumulative_max_fitness = genome.fitness;
                    spec.last_improved_generation = this->generation_num;
                }
            }

            // Add to average
            spec.current_avg_fitness += (IS_ERR(genome.fitness)
                ? 0 // **Provide warning?
                : genome.fitness
            );
        }

        // Correct average fitness
        spec.current_avg_fitness /= (Float_t)(spec.getPopSize());
        
        // Reset allotted offspring
        spec.allotted_offspring = ERR_VAL<GenomeID_t>();
    }
}

void GenePool_s::allotOffspring() {
    Float_t sum_avg_fitnesss = 0;

    for (Species_s& spec : species) {
        // Skip extinct species
        if (spec.extinct) {
            continue;
        }
        
        // Add fitness value
        sum_avg_fitnesss += (IS_ERR(spec.current_avg_fitness)
            ? 0 // **Provide warning?
            : spec.current_avg_fitness
        );
    }

    /*
    The allotted offspring of a species is proportional to its shared
    fitness: `allotted = population_size * shared_fitness / total_shared_fitness`  
    
    The shared fitness of a species is simply the sum of shared fitnesses
    of its members. The shared fitness of an organism is the fitness divided
    by the number of organisms in the species, which is the average fitness of
    the species. However, one caveat:

    On page 13, formula (2) denotes the adjusted fitness function for an organism
    `i` as `f_i`, and defines `f_i` as being the fitness function `f` divided by 
    the summation of all organims whose compatibility distance is below the paramterized
    threshold. However, the author states that this summation "reduces to the number 
    of organisms in the same species as organism i", which is only guaranteed to be
    true if `i` has an identical genome to the representative of the species. 
    
    Imagine a case with a representative `g`, and two very similar genomes `i` and `j`, where
    `i`'s distance from `g` is just below the required threshold, and `j`'s is just 
    above, meaning despite similarity, `i` is placed into `g`'s species and `j` is not. 
    In this case, the distance between `i` and `j` is negligable, yet they are not in the
    same species. Compatiblity distance determines if two organisms are similar enough to
    be in the same species, but not whether or not they actually are. Note that it is 
    explicitly stated that species do not overlap. 
    */
    for (Species_s& spec : species) {
        if (spec.extinct) continue;

        if (sum_avg_fitnesss <= 01e-30) {
            spec.allotted_offspring = PARAMETERS.population_size / (Float_t)(species.size());
        } 
        else {
            spec.allotted_offspring = std::max((Float_t)0,
                PARAMETERS.population_size
                * (spec.current_avg_fitness / sum_avg_fitnesss)
            );
        }

        if (spec.allotted_offspring < 1)
            spec.extinct = true;
    }
}

void GenePool_s::evaluatePopulation() {
    if (has_been_evaluated) return;
    if (EVALUATE_GENOME == nullptr) return;

    for (Genome_s& genome : gene_pool) {
        genome.evaluate();
    }

    has_been_evaluated = true;
}

void GenePool_s::cullSpecies() {
    const GenerationID_t& STAG_LIMIT = PARAMETERS.stagnation.species_stagnation_limit;
    
    auto sitr = species.begin();
    GenerationID_t staleness;

    while (sitr != species.end()) {
        auto& spec = *sitr;
        staleness = this->generation_num - spec.last_improved_generation;

        if (spec.extinct || staleness > STAG_LIMIT || spec.getPopSize() < 1) {
            spec.extinct = true;
            sitr = species.erase(sitr);
        }
        else {
            spec.last_living_generation = this->generation_num;
            ++sitr;
        }
    }
}

void GenePool_s::cullFromSpecies(Species_s& spec) {
    struct SortIDsByFitness_s {
        const std::vector<Genome_s>& POOL;

        inline SortIDsByFitness_s(const std::vector<Genome_s>& POOL)
            : POOL(POOL)
        {}

        bool operator()(const GenomeID_t& lhs, const GenomeID_t& rhs) {
            const Genome_s& lhsg = POOL[lhs];
            const Genome_s& rhsg = POOL[rhs];
            return lhsg.getFitness() > rhsg.getFitness();
        }
    };

    const Float_t& CULL_RATIO = PARAMETERS.reproduction.cull_ratio;

    // Get list of genome IDs sorted by fitness
    std::list<GenomeID_t> sorted_genomes(spec.members.begin(), spec.members.end());
    sorted_genomes.sort(SortIDsByFitness_s(gene_pool));

    const GenomeID_t SURV_COUNT = sorted_genomes.size() * (1.0 - CULL_RATIO);

    // Remove lowest-performers
    auto itr = sorted_genomes.begin();
    std::advance(itr, SURV_COUNT);
    sorted_genomes.erase(itr, sorted_genomes.end());

    // Copy update member list into species
    spec.members.clear();
    spec.members = std::vector<GenomeID_t>(
        sorted_genomes.begin(), 
        sorted_genomes.end()
    );
}

void GenePool_s::cullFromAllSpecies() {
    for (auto& spec : species) {
        if (spec.extinct) continue;
        cullFromSpecies(spec);
    }
}

GenomeID_t GenePool_s::selectFromOtherSpecies(const Species_s& given) {
    GenomeID_t count = 0;

    for (const Species_s& spec : species) {
        if (spec.extinct) continue;
        if (&spec == &given) continue;
        count += spec.getPopSize();
    }

    GenomeID_t rand = RandUtil::randUpTo(count);
    for (const Species_s& spec : species) {
        if (spec.extinct) continue;
        if (&spec == &given) continue;
        if (rand < spec.getPopSize()) return spec.members[rand];

        rand -= spec.getPopSize();
    }

    return ERR_VAL<GenomeID_t>();
}

void GenePool_s::reproduce(
        const Species_s& spec, 
        std::vector<Genome_s>& child_gene_pool
) {
    const Float_t& CROSSOVER_PROPORTION = PARAMETERS.reproduction.crossover_proportion;
    const Float_t& INTERSPECIES_RATE = PARAMETERS.reproduction.crossover.interspecies_mating_rate;
    
    // Validate species
    if (
        spec.extinct 
        || spec.allotted_offspring < 1
        || IS_ERR(spec.allotted_offspring)
        || spec.allotted_offspring > PARAMETERS.population_size
        || spec.getPopSize() < 1
    ) {
        return;
    }

    // Create shuffled version of the ID list
    std::vector<GenomeID_t> member_ids(spec.members);
    GenomeID_t a,b;
    for (GenomeID_t i = 1; i < member_ids.size(); ++i) {
        RandUtil::randUniquePair(a,b,member_ids.size());
        std::swap(member_ids[a], member_ids[b]);
    }

    const bool ALLOW_INTRA_CROSS = spec.getPopSize() > 1;
    const bool ALLOW_INTER_CROSS = species.size() > 1;

    // Indices for `gene_pool`
    GenomeID_t iA=0,iB=0;
    // Indices for `member_ids`
    GenomeID_t miA,miB;
    for (GenomeID_t i = 0; i < spec.allotted_offspring; ++i) {
        miA = i % member_ids.size();
        iA = member_ids[miA];


        // If crossover
        if (0
            || (ALLOW_INTRA_CROSS && RandUtil::randProb(CROSSOVER_PROPORTION))
            || (ALLOW_INTER_CROSS && !ALLOW_INTRA_CROSS && RandUtil::randProb(CROSSOVER_PROPORTION*INTERSPECIES_RATE))
        ) {
            // If only interspecies crossover is allowed, or the a random chance for interspecies has been satisfied
            if (!ALLOW_INTRA_CROSS || RandUtil::randProb(INTERSPECIES_RATE)) {
                // Select randomly from any live species
                iB = selectFromOtherSpecies(spec);
            }
            // Otherwise, intraspecies crossover
            else {
                // Select random other member of this species
                miB = RandUtil::randCompleteUniquePair(miA, member_ids.size());
                iB = member_ids[miB];
            }

            // Add crossover of selected organisms
            child_gene_pool.push_back(
                Genome_s::crossover(
                    gene_pool[iA],
                    gene_pool[iB]
                )
            );
        }
        // If mutation
        else {
            // Add a mutated clone of the selected organism
            child_gene_pool.push_back(
                gene_pool[iA].makeMutatedClone()
            );
        }
    }
}

std::vector<Genome_s> GenePool_s::reproduce() {
    // Create child gene pool
    std::vector<Genome_s> child_gene_pool;
    child_gene_pool.reserve(PARAMETERS.population_size);

    for (auto& spec : species) {
        if (spec.extinct) continue;

        reproduce(
            spec,
            child_gene_pool
        );
    }

    return std::move(child_gene_pool);
}

void GenePool_s::selectRepresentatives() {
    for (Species_s& spec : species) {
        if (spec.extinct) continue;
        spec.representative = RandUtil::randFrom(spec.members);
    }
}

void GenePool_s::takeNewMembers(
    Species_s& spec, 
    std::list<GenomeID_t>& remaining_child_genome_ids,
    const std::vector<Genome_s>& child_gene_pool
) {
    const Genome_s& REP = spec.getRepresentative();
    const Float_t& THRESH = PARAMETERS.cdf.distance_thresh;

    spec.members.clear();

    auto citr = remaining_child_genome_ids.begin();
    Float_t dist;

    while (citr != remaining_child_genome_ids.end()) {
        const Genome_s& child = child_gene_pool[*citr];
        dist = Genome_s::compatibilityDistance(REP, child);

        if (dist < THRESH) {
            spec.members.push_back(*citr);
            citr = remaining_child_genome_ids.erase(citr);
        }
        else {
            ++citr;
        }
    }
}

void GenePool_s::forceSingleSpecies() {
    if (gene_pool.size() < 1) return;
    species.clear();

    Species_s& spec = species.emplace_back(*this);
    spec.members.reserve(gene_pool.size());

    for (GenomeID_t i = 0; i < gene_pool.size(); ++i) {
        spec.members.push_back(i);
    }
}

void GenePool_s::speciate(const std::vector<Genome_s>& child_gene_pool) {
    std::list<GenomeID_t> remaining_child_genome_ids;
    for (GenomeID_t i = 0; i < child_gene_pool.size(); ++i)
        remaining_child_genome_ids.push_back(i);

    // Take members for each existing species
    for (auto& spec : species) {
        if (spec.extinct) continue;
        takeNewMembers(
            spec,
            remaining_child_genome_ids,
            child_gene_pool
        );
    }

    while ( ! remaining_child_genome_ids.empty()) {
        // Create new species with the next child as a representative
        Species_s& new_spec = species.emplace_back(*this);
        new_spec.representative = remaining_child_genome_ids.front();
        new_spec.members.push_back(new_spec.representative);
        remaining_child_genome_ids.pop_front();

        // Take members for the new species
        takeNewMembers(
            new_spec,
            remaining_child_genome_ids,
            child_gene_pool
        );
    }
}

void GenePool_s::newGeneration() {
    // Esnure there are genomes
    if (gene_pool.size() < 1) return;
    // Ensure there is at least one species
    if (species.size() < 1) forceSingleSpecies();

    // Ensure the most recent population has been evaluated and species stats have been generated
    if (!has_been_evaluated) {
        evaluatePopulation();
        updateSpeciesFitnessStats();
    }

    // Cull stale species
    cullSpecies();
    // Cull low-performing organisms
    cullFromAllSpecies();
    // Update stats again
    updateSpeciesFitnessStats();

    // Allot offspring to each species
    allotOffspring();
    // Cull any species marked for extinction (i.e. assigned <1 offspring during `updateSpeciesStats()`)
    cullSpecies();

    // Produce next generation
    auto child_gene_pool = reproduce();
    ++generation_num;

    // Select representatives and get members for each species
    selectRepresentatives();
    speciate(child_gene_pool);
    gene_pool = std::move(child_gene_pool);
    has_been_evaluated = false;

    // Remove any empty species
    cullSpecies();

    // Evaluate the new population
    evaluatePopulation();
    updateSpeciesFitnessStats();
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
    out << "CURRENT GENERATION: " << pool.generation_num << std::endl;
    out << "Population size: " << pool.gene_pool.size() << std::endl;

    out << "Parameters {" << std::endl
        << "\tInputs:  " << pool.INPUT_NODE_COUNT << std::endl
        << "\tOutputs: " << pool.INPUT_NODE_COUNT << std::endl
        << "\tBais:    " << (pool.HAS_BIAS_NODE ? "present" : "absent")
        << std::endl << "}" << std::endl;

    for (const auto& spec : pool.species) {
        out << "Species " << spec.ID << ": {" << std::endl
            << "\tExitinct:        " << (spec.extinct ? "yes" : "no") << std::endl
            << "\tMAX Fitness OAT: " << spec.cumulative_max_fitness << std::endl
            << "\tCurrent Max Fit: " << spec.current_max_fitness<< std::endl
            << "\tCurrent Avg Fit: " << spec.current_avg_fitness << std::endl
            << "\tAppear in gen:   " << spec.generation_of_inception << std::endl
            << "\tLast Imprvd Gen: " << spec.last_improved_generation << std::endl
            << "\tMembers: {" << std::endl;
        for (const auto& gid : spec.members) {
            const auto& genome = pool.gene_pool[gid];
            out << "\t\t";
            genome.simplifiedPrint(out, (genome.getGenomeSize() < 25));
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
    const Float_t& KDC = POOL.PARAMETERS.reproduction.crossover.keep_disabled_connection;

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
            Gene_s& gene = child.genome.back();

            // Chance to disable if enabled
            if (gene.enabled && KDC > 0.5) {
                gene.enabled = RandUtil::randProb(2-2*KDC);
            }
            // Chance to enable if disabled
            else if (!gene.enabled && KDC < 0.5) {
                gene.enabled = RandUtil::randProb(1-2*KDC);
            }
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

Float_t Genome_s::evaluate() {
    if (POOL.EVALUATE_GENOME == nullptr) return ERR_VAL<Float_t>();
    fitness = POOL.EVALUATE_GENOME(*this);
    return fitness;
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

void Genome_s::simplifiedPrint(std::ostream& out, const bool print_genome) const {
    // ID + parents
    out << "Genome " << ID << " (";
    if (!IS_ERR(PARENT_A)) out << PARENT_A;
    if (!IS_ERR(PARENT_B)) out << "x" << PARENT_B;
    out << "): ";

    // Fitness
    out.precision(5);
    out << "Fit: " << fitness << ", {";
    if (print_genome)
        for (const auto& gene : genome)
            out << gene.INNOVATION_NUM << ", ";
    else
        out << "Size: " << genome.size(); 
    out << "}";
}


}