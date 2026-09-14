#include "../inc/NEAT.hpp"

using namespace NEAT;

auto params_more_mutations = DEFAULT_PARAMETERS;

Float_t evaluateNetwork(const Genome_s& genome) {
    return 0;
}

void testMakeGenome(NodeID_t in, NodeID_t out, bool bias) {
    std::cout << "### MAKE GENOME: " << in << ", " << out << ", " << bias << std::endl;


    GenePool_s pool(
        in, out, bias,
        DEFAULT_PARAMETERS,
        evaluateNetwork
    );

    pool.makeGenome(true);

    std::cout << pool.gene_pool.back();
    std::cin.get();
}

// An "eyeball" test, just makes a bunch of genomes and prints them to the console
void testMakeGenome() {
    testMakeGenome(1,1,false);
    testMakeGenome(1,3,false);
    testMakeGenome(3,1,false);
    testMakeGenome(1,1,true);
    testMakeGenome(5,3,true);
    testMakeGenome(20,20,false);
}

void testMakeGenePoolAndMutate(NodeID_t in, NodeID_t out, bool bias, Parameters_s params) {
    GenePool_s pool(
        in, out, bias,
        params,
        evaluateNetwork
    );

    const Genome_s genome1 = pool.makeGenome(true);
    pool.addGenome(genome1, 4);

    std::cout << pool;

    for (size_t i = 0; i < 10; ++i) {
        for (auto& genome : pool.gene_pool) {
            genome.mutate();
        }
    }

    std::cout << pool;
}

int main() {
    params_more_mutations.reproduction.mutation.rates = {
        .add_node = 0.3,
        .add_connection = 1.8,
        .add_bias = 0.5,
        .disable_connection = 0,
        .enable_connection = 0,
        .weight = 0.5,
    };

    // testMakeGenome();
    testMakeGenePoolAndMutate(2,2,true,params_more_mutations);
}