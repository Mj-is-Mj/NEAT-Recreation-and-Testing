#include "../inc/NEAT.hpp"

using namespace NEAT;

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


int main() {
    testMakeGenome();
}