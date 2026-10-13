#include <cstdio>

#define NO_PRIVATIZATION
#include "../inc/NEAT.hpp"

int test_getRandomNodeOfTypes() {
    NEAT::GenePool_s pool_a(3,13,true,NEAT::DEFAULT_PARAMETERS,nullptr);
    NEAT::GenePool_s pool_b(3,13,false,NEAT::DEFAULT_PARAMETERS,nullptr);

    NEAT::Genome_s genome_a(pool_a,true);
    NEAT::Genome_s genome_b(pool_b,true);
    genome_a.mutateAddNode();
    genome_b.mutateAddNode();

    for (NEAT::NodeTypes_t t = 0; t <= NEAT::NodeType_e::ALL; ++t) {
        if (t == NEAT::NodeType_e::ERROR) continue;

        printf("Testing for case types=0x%1X... ", t);

        for (NEAT::NodeID_t i = 0; i < genome_a.node_count*0x3FFF; ++i) {

            NEAT::NodeID_t n_a = genome_a.getRandomNodeOfTypes(t);
            NEAT::NodeID_t n_b = genome_b.getRandomNodeOfTypes(t);

            NEAT::NodeType_e t_a = pool_a.getNodeType(n_a);
            NEAT::NodeType_e t_b = pool_b.getNodeType(n_b);

            assert(t_a & t);
            if (t == NEAT::NodeType_e::BIAS) {
                assert(NEAT::IS_ERR(n_b));
                assert(t_b == NEAT::NodeType_e::ERROR);
            }
            else {
                assert(t_b & t);
            }

            assert(NEAT::IS_ERR(n_a) || (0 <= n_a && n_a < genome_a.node_count));
            assert(NEAT::IS_ERR(n_b) || (0 <= n_b && n_b < genome_a.node_count));
        }

        printf("Pass\n");
    }

    return 0;
}

int main() {
    test_getRandomNodeOfTypes();
}