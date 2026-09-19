
#include "../inc/NEAT.hpp"
#include <cmath>
using namespace NEAT;

constexpr Parameters_s LOCAL_PARAMS {
    .reproduction = {
        .mutation = {
            .rates = {
                .add_node = 0.8,
                .add_connection = 0.8,
                .add_bias = 0.5,
                .disable_connection = 0.5,
                .enable_connection = 0.5,
                .weight = 0.5,
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
        .crossover_proportion = 0.5,
    },
    .stagnation = {
        .species_stagnation_limit = 15,
        .population_stagnation_limit = 20,
        .minimum_species_count = 2,
    },
    .cdf = {
        .c1=1.0, .c2=1.0, .c3=0.4,
        .distance_thresh = 0.7,
    },
    .population_size = 50,
};

Float_t evaluateNetwork(const Genome_s& genome) {
    static constexpr Float_t A = 30;

    Float_t connections = 0;
    Float_t weights = 0;
    for (const auto& gene : genome.genome) {
        if (!gene.enabled) continue;
        weights += std::abs(gene.weight);
        connections += 1;
    }
    weights = std::sqrt(std::abs(weights));

    #define EVAL(x,w) ((2*A*x) / ((x*x + A*A)*w + 0.1))
    return connections / (10*weights + 0.1);
    #undef EVAL
}

int main() {
    GenePool_s pool(
        3, 2, true, 
        LOCAL_PARAMS, 
        evaluateNetwork
    );

    const Genome_s BASE(pool, true);
    pool.addGenome(BASE, 3);
    std::cout << pool;


    for (size_t i = 0; i < 100; ++i) {
        pool.newGeneration();

        std::cout << pool << std::endl;
    }

    pool.cullFromAllSpecies();
    pool.updateSpeciesFitnessStats();

    std::cout << pool << std::endl;
}