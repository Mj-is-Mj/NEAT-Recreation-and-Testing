
#include "../inc/NEAT.hpp"
#include "../inc/randutil.hpp"
using namespace NEAT;

constexpr Parameters_s LOCAL_PARAMS {
    .reproduction = {
        .mutation = {
            .rates = {
                .add_node = 0.3,
                .add_connection = 1.8,
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
    .population_size = 10,
};

Float_t evaluateNetwork(const Genome_s& genome) {
    static constexpr Float_t A = 30;
    const Float_t x = genome.getGenomeSize();
    const Float_t fit = (2*A*x) / (x*x + A*A);

    return 10*fit*fit;
}

int main() {
    GenePool_s pool(
        3, 2, true, 
        LOCAL_PARAMS, 
        evaluateNetwork
    );

    const Genome_s BASE(pool, true);
    pool.addGenome(BASE, 12);

    pool.evaluatePopulation();
    std::cout << pool;


    for (size_t i = 0; i < 8; ++i) {
        // Mutate all
        for (auto& genome : pool.gene_pool) {
            genome.mutate();
        }

        // Select pair near end of gene pool
        GenomeID_t a,b;
        RandUtil::randUniquePair(a,b,GenomeID_t{4});
        a += pool.gene_pool.size() - 4;
        b += pool.gene_pool.size() - 4;

        // Crossover pair and add to end
        pool.addGenome(
            Genome_s::crossover(
                pool.gene_pool[a],
                pool.gene_pool[b]
            )
        );
    }

    pool.has_been_evaluated = false;
    pool.evaluatePopulation();
    std::cout << pool;

}