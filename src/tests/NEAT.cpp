#include "../inc/NEAT.hpp"
#include "randutil.hpp"
#include <cstdlib>

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

void validateCrossover(const Genome_s& pA, const Genome_s& pB, const Genome_s& child) {

    // Get all parental innovations
    std::list<GenomeID_t> parental_innovations;
    for (const auto& gene : pA.genome) {
        parental_innovations.push_back(gene.INNOVATION_NUM);
    }
    for (const auto& gene : pB.genome) {
        parental_innovations.push_back(gene.INNOVATION_NUM);
    }
    parental_innovations.sort();
    parental_innovations.unique();

    // Get all innovations in the child
    std::list<GenomeID_t> child_innovations;
    for (const auto& gene : child.genome) {
        child_innovations.push_back(gene.INNOVATION_NUM);
    }

    // Both innovations lists should be equal
    auto pitr = parental_innovations.begin();
    auto citr = child_innovations.begin();
    while (pitr != parental_innovations.end() && citr != child_innovations.end()) {
        if (*pitr != *citr) {
            std::cout << "ERROR: Parent innovations differ from child innovations" << std::endl;
            std::cout << "Parent: {";
            for (const auto& p : parental_innovations) std::cout << p << ", ";
            std::cout << "}" << std::endl;
            std::cout << "Child: {";
            for (const auto& c : child_innovations) std::cout << c << ", ";
            std::cout << "}" << std::endl;
            exit(EXIT_FAILURE);
        }

        ++citr;
        ++pitr;
    }

    if (pitr != parental_innovations.end() || citr != child_innovations.end()) {
        std::cout << "ERROR: Parent innovations differ from child innovations" << std::endl;
        std::cout << "Parent: {";
        for (const auto& p : parental_innovations) std::cout << p << ", ";
        std::cout << "}" << std::endl;
        std::cout << "Child: {";
        for (const auto& c : child_innovations) std::cout << c << ", ";
        std::cout << "}" << std::endl;
        exit(EXIT_FAILURE);
    }


    // Check innovation number order of the child's genome
    GeneID_t prev_innov = 0;
    for (const auto& gene : child.genome) {
        if (prev_innov < gene.INNOVATION_NUM) {
            prev_innov = gene.INNOVATION_NUM;
            continue;
        }

        if (prev_innov == gene.INNOVATION_NUM && prev_innov == 0) {
            continue;
        }

        std::cout << child.POOL;

        child.simplifiedPrint(std::cout); std::cout << std::endl;
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
}

void testCrossover(NodeID_t in, NodeID_t out, bool bias, Parameters_s params) {
    GenePool_s pool(
        in, out, bias,
        params,
        evaluateNetwork
    );

    const Genome_s genome1 = pool.makeGenome(true).clone();
    pool.addGenome(genome1, 4);

    std::cout << pool;

    for (size_t i = 0; i < 10; ++i) {
        for (auto& genome : pool.gene_pool) {
            genome.mutate();
        }
    }

    std::cout << pool;

    for (size_t i = 0; i < 10; ++i) {
        const Genome_s A = pool.gene_pool[RandUtil::randUpTo(pool.gene_pool.size())];
        const Genome_s B = pool.gene_pool[RandUtil::randUpTo(pool.gene_pool.size())];
        const Genome_s CHILD = Genome_s::crossover(A,B);

        validateCrossover(A,B,CHILD);

        pool.addGenome(CHILD);
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
    // testMakeGenePoolAndMutate(2,2,true,params_more_mutations);
    testCrossover(2,2,true,params_more_mutations);
}