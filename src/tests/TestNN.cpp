#include "../inc/NN.hpp"
#include "../inc/NEAT.hpp"
#include "../inc/mathutil.hpp"
#include "../inc/poles.hpp"

#include <cstdlib>

static_assert(__cplusplus >= 202002L, "TestNN.cpp requires C++20 or greater");
NEAT::Parameters_s LOCAL_PARAMS = NEAT::DEFAULT_PARAMETERS;

bool dbg = false;
// Fitness function that uses a NN:BasicNetwork_s
// Network outputs > 0.5 are treated as 1, else 0
NEAT::Float_t xor_eval(const NEAT::Genome_s& genome) {
    NN::SequentialNetwork_s network(
        genome, 
        NN::Activations::sigmoidX<NN::Float_t{4.9}>,
        NN::Activations::sigmoidX<NN::Float_t{4.9}>
    );

    NN::Float_t inputs[3] = {0,0,1.f}; // inputs[2] is never modified (bias node)

    NN::Float_t fit = 0;
    NN::Float_t result = 0;
    NN::Float_t min = .5f;
    NN::Float_t max = -.5f;
    bool failed_any = false;

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

NEAT::Float_t pole_eval(const NEAT::Genome_s& genome) {
    // NN from genome
    NN::SequentialNetwork_s network(
        genome, 
        NN::Activations::sigmoidX<NN::Float_t{4.9}>,
        NN::Activations::sigmoidX<NN::Float_t{4.9}>
    );

    // Pole balance using default parameters
        // See `PoleProblems::CartWithPoles_s::DEFAULT_PARAMETERS`
    PoleProblems::CartWithPoles_s pole_balance{};

    // Constants
    static constexpr NN::Float_t balanced_thresh = 36 * Math::Constants<NN::Float_t>::PI / 180;
    static constexpr int max_steps = 1e5;

    // Make input array
    NN::Float_t* input_arr = (NN::Float_t*)calloc(genome.getInputNodeCount(), sizeof(NN::Float_t));
    if (input_arr == nullptr) {
        std::cerr << "FAILED TO MALLOC IN " << __func__ << std::endl;
        exit(-1);
    }

    // Loop variables
    int steps = 0;
    NN::Float_t force = 0;
    bool not_failed = true;

    // Main loop
    while(not_failed && steps < max_steps) {
        // Rescale input values to range of [-1, 1]
        #define SCALE(x) (2.*NN::Activations::sigmoid(x) - 1.)
        #define LIN_SCALE(x,r) (x * (1. / r))
        {
            const auto& pp_state = pole_balance.getState();

            // Bias node
            input_arr[0] = 1.f;

            // Cart state
            input_arr[1] = LIN_SCALE(pp_state.cart.x, pole_balance.PARAMETERS.cart.track_range); // **Use a linear scale for this?
            input_arr[2] = SCALE(pp_state.cart.dx);

            // Pole states
            for (size_t i = 0; i < pp_state.poles.size(); ++i) {
                input_arr[3 + 2*i] = SCALE(pp_state.poles[i].theta); // **Use a linear scale for this?
                input_arr[4 + 2*i] = SCALE(pp_state.poles[i].dtheta);
            }
        }
        
        // Eval network
        force = 20. * (*network.stepThis(input_arr) - 0.5);
        
        // Step simulation with force applied (twice)
        pole_balance.step(force);
        pole_balance.step(force);

        {
            const auto& pp_state = pole_balance.getState();

            // If any pole is "out of balance", then fail/end
            for (size_t i = 0; i < pp_state.poles.size(); ++i) {
                if (Math::abs(pp_state.poles[i].theta) > balanced_thresh) {
                    not_failed = false;
                    break;
                }
            }

            // If cart is out of bounds, then fail/end
            if (Math::abs(pp_state.cart.x) > pole_balance.PARAMETERS.cart.track_range)
                not_failed = false;
        }
        ++steps;
    }

    // ** Temporary
    if (steps == max_steps) {
        std::cout << "SOLUTION FOUND" << std::endl;
    }

    free(input_arr);

    return (NEAT::Float_t)steps * 0.02;
}

void test_xor() {
    // srand(time(0));

    NEAT::GenePool_s pool(
        3, 1, 0,
        LOCAL_PARAMS, 
        xor_eval
    );

    const NEAT::Genome_s BASE(pool, true);
    pool.addGenome(BASE, 3);
    pool.evaluatePopulation();
    std::cout << pool;


    for (size_t i = 0; i < 22; ++i) {
        pool.newGeneration();

        std::cout << pool << std::endl;
    }

    std::cout << pool << std::endl;

    dbg = true;

    for (const auto& genome : pool.getGenePool()) {
        pool.evaluateGenome(genome);
    }
}

/*
    Difference in results:
        Actual performance:
            Poles are only being balanced for a few seconds after a hundred or 
            so generations, which is significantly worse than the paper's results. 
        Speciation:
            Figure 7 clearly shows that the population would speciate quite a lot,
            (ending with 14 species), whereas I'm only single single-digit total
            species, with only one surviving. 
*/
void test_pole() {
    // srand(time(0));

    auto params = NEAT::DEFAULT_PARAMETERS;

    const size_t num_inputs = (0
        + 1 // Bias node
        + 2 // Cart position/velocity
        // Pole positions/velocities
        + 2*PoleProblems::CartWithPoles_s::DEFAULT_PARAMETERS.pole_count()
    );

    NEAT::GenePool_s pool(
        num_inputs, 1, 0,
        params, 
        pole_eval
    );

    const NEAT::Genome_s BASE(pool, true);
    pool.addGenome(BASE, 3);
    pool._forceSingleSpecies();
    pool.evaluatePopulation();
    pool.simplifiedPrint(std::cout);


    for (size_t i = 0; i < 256; ++i) {
        pool.newGeneration();

        pool.simplifiedPrint(std::cout);
    }

    pool.simplifiedPrint(std::cout);

    dbg = true;

    for (const auto& genome : pool.getGenePool()) {
        pool.evaluateGenome(genome);
    }
}


int main() {
    test_pole();
    return 0;
}