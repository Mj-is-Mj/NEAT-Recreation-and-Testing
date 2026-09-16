#pragma once

#include "randutil.hpp"
#include <cstdlib>
#include <cassert>

#include <iostream>
#include <ostream>
#include <vector>
#include <list>

namespace NEAT {

struct Parameters_s;
struct GenePool_s;
struct Species_s;
struct Gene_s;
struct Genome_s;

    // TYPEDEFS: Used to avoid ambiguity of different number types
    // E.g. `NodeID_t` will only ever be used in relation to nodes, 
    // and no other data types will be used for identifying or counting 
    // nodes in any capacity. 

// Preferred floating point number type
typedef float   Float_t;
// Node identifier type
typedef size_t  NodeID_t;
// Gene identifier type
typedef size_t  GeneID_t;
// Genome identifier type
typedef size_t  GenomeID_t;
// Species identifier type
typedef size_t  SpeciesID_t;
// Generation identifier type
typedef size_t  GenerationID_t;

// Returns the value denoted as an error for the above typedefs
template<typename T>
inline constexpr T ERR_VAL() { return ~(T{0}); }
// ERR_VAL specifically for Float_t
template<>
inline constexpr Float_t ERR_VAL() { return (Float_t)(-1e38); };
// Checks if value is an error
template<typename T>
inline constexpr bool IS_ERR(const T val) { return val == ERR_VAL<typeof(T)>(); }

// Evaluation function for genomes
typedef Float_t (*EvaluateFunc_t)(const Genome_s&);

// Types of nodes
enum NodeType_e {
    INPUT,
    BIAS,
    OUTPUT,
    HIDDEN,
    ERROR,
};

// Parameters for `GenePool`
struct Parameters_s {
    struct Reproduction_s {
        struct Mutation_s {
            struct Rates_s {
                Float_t add_node;
                Float_t add_connection;
                Float_t add_bias;
                Float_t disable_connection;
                Float_t enable_connection;
                Float_t weight;
            };

            // Rates for each mutation type
            Rates_s rates;
            // The range [-N,N] in which a weight's value can be set to
            Float_t weight_random_range;
            // The probability of a weight being perturbed instead of randomly set
            Float_t weight_perturb_chance;
            // The amount by which a weight can be perturbed
            Float_t weight_perturb_amount;
        };

        struct Crossover_s {
            // Probability for a connection to be disabled if the corresponding gene is disabled in either parent. 
            // This only applies to matching genes. 
            Float_t keep_disabled_connection;
            // The probability of crossing with a member of a different species
            Float_t interspecies_mating_rate;
        };

        // Mutation-related paramters
        Mutation_s mutation;
        // Crossover-related parameters
        Crossover_s crossover;
        // What ratio of the population is culled before reproduction
        Float_t cull_ratio;
        // What proportion of new generations created via crossover as apposed to mutation
        Float_t crossover_proportion;
    };

    struct CDF_s {
        // Constants for the Compatibility Disatnce Function
        Float_t c1, c2, c3;
        // Distance under which two genomes can be considered part of the same species
        Float_t distance_thresh;
    };

    struct Stagnation_s {
        GenerationID_t species_stagnation_limit;
        GenerationID_t population_stagnation_limit;
        GenerationID_t minimum_species_count;
    };

    // Reproduction-related paramters
    Reproduction_s reproduction;
    // Stagnation-related paramters
    Stagnation_s stagnation;
    // Paramters relating to the Compatibility Distance Function
    CDF_s cdf;
    // The initial size of each generation
    SpeciesID_t population_size;
};

// Parameters as defined in the paper
constexpr Parameters_s DEFAULT_PARAMETERS {
    .reproduction = {
        .mutation = {
            .rates = {
                .add_node = 0.03,
                .add_connection = 0.05,
                .add_bias = 0,
                .disable_connection = 0,
                .enable_connection = 0,
                .weight = 0.8,
            },
            .weight_random_range = 2,
            .weight_perturb_chance = 0.9,
            .weight_perturb_amount = 0.1,
        },
        .crossover = {
            .keep_disabled_connection = 0.75,
            .interspecies_mating_rate = 0.001,
        },
        .cull_ratio = 0.5,
        .crossover_proportion = 0.75,
    },
    .stagnation = {
        .species_stagnation_limit = 15,
        .population_stagnation_limit = 20,
        .minimum_species_count = 2,
    },
    .cdf = {
        .c1=1.0, .c2=1.0, .c3=0.4,
        .distance_thresh = 3.0,
    },
    .population_size = 150,
};

// Contains a set of genomes, handles the transition from one generation to the next
struct GenePool_s {
    /// Topology information ///
    const NodeID_t INPUT_NODE_COUNT;
    const NodeID_t OUTPUT_NODE_COUNT;
    const bool HAS_BIAS_NODE;

    /// Parameters ///
    const Parameters_s PARAMETERS;

    /// Evaluation ///
    // Function used to evaluate a genome
    const EvaluateFunc_t EVALUATE_GENOME;
    // Whether or not the current generation has been evaluated
    bool has_been_evaluated;
    // Number of generations since any improvment
    GenerationID_t staleness;

    /// Population ///
    // All genomes in the current generation
    std::vector<Genome_s> gene_pool;
    // List of all species
    std::list<Species_s> species;

    /// Mutables ///
    mutable GeneID_t innovation_num;
    mutable GenomeID_t genome_num;
    mutable SpeciesID_t species_num;
    mutable GenerationID_t generation_num;
    inline GeneID_t getNextInnovationNumber() const { return innovation_num++; }
    inline GeneID_t getNextGenomeNumber() const { return genome_num++; }
    inline GeneID_t getNextSpeciesNumber() const { return species_num++; }
    inline GeneID_t getNextGenerationNumber() const { return generation_num++; }


    /// Constructors ///
    inline GenePool_s(
        const NodeID_t inputs, const NodeID_t outputs, const bool has_bias,
        const Parameters_s parameters, const EvaluateFunc_t eval_func
    )   : INPUT_NODE_COUNT(inputs)
        , OUTPUT_NODE_COUNT(outputs)
        , HAS_BIAS_NODE(has_bias)
        , PARAMETERS(parameters)
        , EVALUATE_GENOME(eval_func)
        , has_been_evaluated(false)
        , staleness(0)
        , innovation_num(0)
        , genome_num(0)
        , species_num(0)
        , generation_num(0)
        , gene_pool()
        , species()
    { gene_pool.reserve(parameters.population_size); }


    /// Initialization ///
    // Clears all genomes and sepcies, resets `innovation_num` and other mutables, etc.
    void clear();
    // Create a genome in the pool and return a reference to it
    // Node: This genome is contained in a vector, so the reference will become invalid on resizes
    inline Genome_s& makeGenome(const bool fully_connect = false, const GenomeID_t count = 1) 
        { return gene_pool.emplace_back(*this, fully_connect); }
    // Add a genome to the pool
    bool addGenome(const Genome_s genome, const GenomeID_t count = 1);

    
    /// Nodes ///
    // Node start/end
    constexpr NodeID_t getInputNodeStart() const { return 0; };
    constexpr NodeID_t getInputNodeEnd() const {return INPUT_NODE_COUNT; };
    constexpr NodeID_t getOutputNodeStart() const { return getInputNodeEnd() + (HAS_BIAS_NODE ? 1 : 0); };
    constexpr NodeID_t getOutputNodeEnd() const { return getOutputNodeStart() + OUTPUT_NODE_COUNT; };
    constexpr NodeID_t getBiasNode() const { return HAS_BIAS_NODE ? getInputNodeEnd() : ERR_VAL<NodeID_t>(); };
    // Random getters
    NodeID_t getRandomInputNodeID() const;
    NodeID_t getRandomOutputNodeID() const;
    // Other node stuff
    constexpr NodeID_t getPredefNodesEnd() const { return getOutputNodeEnd(); };
    constexpr bool isInputNode(const NodeID_t n) const 
        { return getInputNodeStart() <= n && n < getInputNodeEnd(); }
    constexpr bool isOutputNode(const NodeID_t n) const 
        { return getOutputNodeStart() <= n && n < getOutputNodeEnd(); }
    constexpr bool isBiasNode(const NodeID_t n) const
        { return !IS_ERR(n) && n == getBiasNode(); }
    NodeType_e getNodeType(const NodeID_t n) const;

    /// DEBUGGING ///
    friend std::ostream& operator<<(std::ostream& out, const GenePool_s& pool);
    void printNode(std::ostream& out, const NodeID_t n) const;

    /// CULLING ///
    void evaluatePopulation();
    void updateSpeciesStats();
    void cullStaleSpecies();
    void cullFromSpecies(Species_s& spec);
    void cullFromSpecies();

    // REPRODUCTION AND SPECIATION///
    void reproduce(
        const Species_s& spec, 
        const std::vector<GenomeID_t>& interspecies_pool,
        std::vector<Genome_s>& child_gene_pool
    );
    std::vector<Genome_s> reproduce();
    void selectRepresentatives();
    void takeNewMembers(
        Species_s& spec, 
        std::list<GenomeID_t>& remaining_child_genome_ids,
        const std::vector<Genome_s>& child_gene_pool
    );
    // If there are no species, add all genomes to a new species
    void forceSpeciate();
    void speciate(const std::vector<Genome_s>& child_gene_pool);

    /// "MAIN" FUNCTION ///
    void newGeneration();
};

// A single species within a `GenePool`
struct Species_s {
    const GenePool_s& POOL;
    const SpeciesID_t ID;
    // The last generation when the species was alive and evaluated
    GenerationID_t last_living_generation;
    // The maximum fitness seen in the species across all generations
    Float_t cumulative_max_fitness;
    // The maximum fitness of the current generation
    Float_t current_max_fitness;
    // The average fitness of the current generation
    // This is also the shared fitness of the species
    Float_t current_avg_fitness;
    // The number of offspring allocated to this species
    GenomeID_t allotted_offspring;
    // How many generations since the last imrpovement to `cumulative_max_fitness`
    GenerationID_t staleness;
    // Representative when speciating (index for `POOL.gene_pool`)
    GenomeID_t representative;
    // Self-explanitory
    bool extinct;

    // The IDs of the genomes of all members of the species
    std::vector<GenomeID_t> members;

    template<typename... GenomeID_tmp>
    inline Species_s(const GenePool_s& POOL, const GenomeID_tmp... genome_ids)
            : POOL(POOL)
            , ID(POOL.getNextSpeciesNumber())
            , last_living_generation(POOL.generation_num)
            , cumulative_max_fitness(0)
            , current_max_fitness(0)
            , current_avg_fitness(0)
            , staleness(0)
            , representative(ERR_VAL<GenomeID_t>())
            , members{genome_ids...}
            , extinct(false)
    {}

    inline Species_s(const Species_s& other)
            : POOL(other.POOL)
            , ID(other.ID)
            , last_living_generation(other.last_living_generation)
            , cumulative_max_fitness(other.cumulative_max_fitness)
            , current_max_fitness(other.current_max_fitness)
            , current_avg_fitness(other.current_avg_fitness)
            , staleness(other.staleness)
            , representative(other.representative)
            , members(other.members)
            , extinct(other.extinct)
    {}

    inline GenomeID_t getPopSize() const 
        { return members.size(); }
    inline GenomeID_t getRandomGenomeID() const 
        { return members[RandUtil::randUpTo(members.size())]; }
    inline void getRandomGenomeIDPair(GenomeID_t& a, GenomeID_t& b) const 
        { return RandUtil::randUniquePair(a,b,members.size()); }
    inline const Genome_s& getRepresentative() const
        { return POOL.gene_pool[representative]; }
};

// A single gene representing a connection
struct Gene_s {
    const GeneID_t INNOVATION_NUM;
    const NodeID_t FROM, TO;
    Float_t weight;
    bool enabled;

    inline Gene_s(
        const GeneID_t innovation_num, 
        const NodeID_t from, const NodeID_t to,
        const Float_t weight, const bool enabled
    )   : INNOVATION_NUM(innovation_num)
        , FROM(from)
        , TO(to)
        , weight(weight)
        , enabled(enabled)
    {}

    inline Gene_s(const Gene_s& other)
        : INNOVATION_NUM(other.INNOVATION_NUM)
        , FROM(other.FROM)
        , TO(other.TO)
        , weight(other.weight)
        , enabled(other.enabled)
    {}

    inline bool operator==(const Gene_s& other) const
        { return this->INNOVATION_NUM == other.INNOVATION_NUM;  }
    
    friend std::ostream& operator<<(std::ostream& out, const Gene_s& gene);
};

// A genome, used to construct a specimen/network
struct Genome_s {
    // The `GenePool_s` that this genome is contained in
    const GenePool_s& POOL;
    const GenomeID_t ID;
    const GenomeID_t PARENT_A, PARENT_B; // The IDs of each parent
    NodeID_t node_count;
    Float_t fitness;
    std::vector<Gene_s> genome;


    /// CONSTRUCTORS ///
    
    // Copy constructor
    Genome_s(const Genome_s& other);
    // Parameter-based constructor
    Genome_s(
        const GenePool_s& pool, const GenomeID_t& ID, 
        const GenomeID_t parent_a, const GenomeID_t parent_b,
        const NodeID_t node_count, const std::vector<Gene_s>& genome
    );
    // Create a empty or fully-connected genome
    Genome_s(const GenePool_s& pool, const bool fully_connect = false);
    
    // Returns a genome that is an identical clone of another genome. 
    // Distinct from the copy constructor since yields a different ID. 
    // Result is considered a child of the genome it's cloned from. 
    inline Genome_s clone() const
        { return Genome_s(POOL, POOL.getNextGenomeNumber(), ID, ERR_VAL<NodeID_t>(), node_count, genome); }
    
    // Returns a genome that is an identical child of another genome. 
    // Distinct from the copy constructor since yields a different ID. 
    // Result is considered a sibiling of the genome it's cloned from. 
    inline Genome_s duplicate() const
        { return Genome_s(POOL, POOL.getNextGenomeNumber(), PARENT_A, PARENT_B, node_count, genome); }


    /// GETTERS ///

    // Counts/sizes
    constexpr bool hasBias() const { return POOL.HAS_BIAS_NODE; }; 
    constexpr NodeID_t getNodeCount() const { return node_count; };
    constexpr NodeID_t getInputNodeCount() const { return POOL.INPUT_NODE_COUNT; };
    constexpr NodeID_t getOutputNodeCount() const { return POOL.OUTPUT_NODE_COUNT; };
    constexpr NodeID_t getHiddenNodeCount() const 
        { return node_count - getInputNodeCount() - getOutputNodeCount(); };
    inline GeneID_t getGenomeSize() const { return genome.size(); }
    
    // Identity
    constexpr GenomeID_t getID() const { return ID; };
    const std::vector<Gene_s>& getGenome() const { return genome; };
    constexpr const GenePool_s& getGenePool() const { return POOL; };
    constexpr Float_t getFitness() const { return fitness; };
    
    // Random getters
    NodeID_t getRandomNodeID() const;
    NodeID_t getRandomHiddenNodeID() const;
    NodeID_t getRandomInputOrHiddenNodeID() const;
    NodeID_t getRandomOutputOrHiddenNodeID() const;
    GeneID_t getRandomGeneID() const;
    inline Gene_s& getRandomGenome() { return genome[getRandomGeneID()]; }
    
    // Checks and whatnot
    bool connectionExists(const NodeID_t from, const NodeID_t to);
    

    /// MODIFICIATION ///
    void addConnection(
        const NodeID_t from, const NodeID_t to, 
        const Float_t weight = 1, const bool enabled = true
    );
    void addNode(Gene_s& connection);

    /// MUTATIONS ///
    inline Float_t getRandomWeight() const { 
        return (RandUtil::randF<Float_t>()-0.5)
        *(2*POOL.PARAMETERS.reproduction.mutation.weight_random_range);
    }
    inline Float_t getRandomWeightPerturbation() const { 
        return (RandUtil::randF<Float_t>()-0.5)
        *(2*POOL.PARAMETERS.reproduction.mutation.weight_perturb_amount);
    }
    void mutateAddNode();
    bool mutateAddConnection();
    bool mutateAddBias();
    bool mutateSetConnection(const bool enabled);
    void mutateSetRandomWeight();
    void mutatePerturbWeight();
    void mutate();

    /// REPRODUCTION ///
    static Genome_s makeMutatedClone(const Genome_s& parent);
    static Genome_s crossover(const Genome_s& pA, const Genome_s& pB);

    /// METRICS ///
    static Float_t compatibilityDistance(const Genome_s& A, const Genome_s& B);
    Float_t evaluate();
    
    /// ALIASES ///
    // Calls to `GenePool` member functions
    inline NodeID_t getRandomInputNodeID() const { return POOL.getRandomInputNodeID(); };
    inline NodeID_t getRandomOutputNodeID() const { return POOL.getRandomOutputNodeID(); };
    inline NodeID_t getInputNodeStart() const { return POOL.getInputNodeStart(); };
    inline NodeID_t getInputNodeEnd() const { return POOL.getInputNodeEnd(); };
    inline NodeID_t getOutputNodeStart() const { return POOL.getOutputNodeStart(); };
    inline NodeID_t getOutputNodeEnd() const { return POOL.getOutputNodeEnd(); };
    inline NodeID_t getHiddenNodeStart() const { return POOL.getPredefNodesEnd(); };
    inline NodeID_t getBiasNode() const { return POOL.getBiasNode(); };
    inline bool isInputNode(const NodeID_t n) const { return POOL.isInputNode(n); }
    inline bool isOutputNode(const NodeID_t n) const { return POOL.isOutputNode(n); }
    // Misc
    inline NodeID_t getHiddenNodeEnd() const { return node_count; };
    inline GeneID_t getConnectionCount() const { return getGenomeSize(); };
    inline GeneID_t getRandomConnectionID() const { return getRandomGeneID(); }
    inline void addBias(const NodeID_t to, const Float_t weight = 1, const bool enabled = true)
        { assert(hasBias()); addConnection(getBiasNode(), to, weight, enabled); }
    inline void mutateEnableConnection() { mutateSetConnection(true); };
    inline void mutateDisableConnection() { mutateSetConnection(false); };
    inline Genome_s makeMutatedClone() { return makeMutatedClone(*this); }
    
    /// DEBUGGING ///
    friend std::ostream& operator<<(std::ostream& out, const Genome_s& genome);
    void simplifiedPrint(std::ostream& out) const;
    inline void printNode(std::ostream& out, const NodeID_t n) const { POOL.printNode(out, n); }
};


}