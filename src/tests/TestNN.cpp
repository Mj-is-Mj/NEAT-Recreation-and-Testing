#include "../inc/NN.hpp"
#include "../inc/NEAT.hpp"

NEAT::Parameters_s LOCAL_PARAMS = NEAT::DEFAULT_PARAMETERS;

bool dbg = false;

// Fitness function that uses a NN:BasicNetwork_s
// Network outputs > 0.5 are treated as 1, else 0
NEAT::Float_t xor_eval(const NEAT::Genome_s& genome) {
    NN::SequentialNetwork_s network(
        genome, 
        NN::Activations::sigmoid,
        NN::Activations::sigmoid
    );

    NN::Float_t inputs[3] = {0,0,1.f}; // inputs[2] is never modified (bias node)

    NN::Float_t fit = 0;
    NN::Float_t result = 0;
    NN::Float_t min = .5f;
    NN::Float_t max = -.5f;
    bool failed_any = false;

    if (genome.getID() == 2376)
        min += 0;

    // For each possible input
    for (size_t i = 0; i < 4; ++i) {
        // Set inputs
        inputs[0] = (i&1 ? 1.f : 0.f);
        inputs[1] = (i&2 ? 1.f : 0.f);

        // Evaluate network, map result from [0,1] to [-0.5, 0.5]
        result = *network.stepThis(inputs) - NN::Float_t{0.5};

        if (((i&1) ^ (i&2)) == 0) result = -result;

        min = std::min(min, result);
        max = std::max(max, result);

        // Fit increments by at most +/- 0.5
        fit += result;
        failed_any |= result < 0;
    }

    if (!failed_any) {
        if (dbg) printf("ID %5ld: %s: ", genome.getID(), (failed_any ? "__fail" : "PASSED"));
        if (dbg) printf("MIN = %4.2f | MAX = %4.2f\n", min, max);
    }

    // Map fit from [-2,2] to [0,1] because of some bugs with negative fitness values that I'll fix soon
    return (fit + 2.0f)*0.25f;
};


int main() {
    // srand(time(0));

    NEAT::GenePool_s pool(
        3, 1, 0,
        LOCAL_PARAMS, 
        xor_eval
    );

    const NEAT::Genome_s BASE(pool, true);
    pool.addGenome(BASE, 3);
    std::cout << pool;


    for (size_t i = 0; i < 22; ++i) {
        pool.newGeneration();

        std::cout << pool << std::endl;
    }

    pool.cullFromAllSpecies();
    pool.updateSpeciesFitnessStats();

    std::cout << pool << std::endl;

    dbg = true;

    for (auto& genome : pool.gene_pool) {
        genome.evaluate();
    }
}