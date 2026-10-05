#include "../inc/NEAT.hpp"

#include <glm/glm.hpp>
#include <SDL3/SDL.h>

using namespace NEAT;

namespace glm { 
    typedef glm::vec<2,int,glm::qualifier::defaultp> vec2i;
    int doti(const vec2i a, const vec2i b);

    inline int doti(const vec2i v, const int x, const int y) { return v.x*x + v.y*y; }

    inline int lengthi(const vec2i v) { return std::sqrt(v.x*v.x + v.y*v.y); }
}

void arrangeGraph(const Genome_s& genome, std::vector<glm::vec2>& positions, const size_t iterations = 256);

void drawSquare(SDL_Surface* img, const glm::vec2i lower, const glm::vec2i upper, const Uint32 color);

void drawCircle(SDL_Surface* img, const glm::vec2i center, const int rad, const Uint32 color);

void drawLine(SDL_Surface* img, const glm::vec2i p0, const glm::vec2i p1, const Uint32 color, const int thick = 1);

void drawTriangle(SDL_Surface* img, const glm::vec2i A, const glm::vec2i B, const glm::vec2i C, const Uint32 color);

void drawArrow(SDL_Surface* img, const glm::vec2i start, const glm::vec2i end, const int line_width, const int arrow_size, const Uint32 color);

void drawGraph(SDL_Surface* img, const Genome_s& genome, std::vector<glm::vec2>& positions);
