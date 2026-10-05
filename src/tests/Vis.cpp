#include "../inc/GraphVis.hpp"
#include "NEAT.hpp"
#include <SDL3/SDL_blendmode.h>
#include <SDL3/SDL_surface.h>
using namespace NEAT;


#define WINDOW_W    960
#define WINDOW_H    540
#define MAX_FPS     144

constexpr Uint64 frametime_ns = 1e9 / MAX_FPS;

int main() {
    // SDL Init
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "ERROR: SDL_Init failed");
        SDL_Log("%s", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    // Create window
    SDL_Window* win = SDL_CreateWindow(
        "SDL Window", 
        WINDOW_W, WINDOW_H,
        0
    );

    // Display error and exit if window creation failed
    if (win == nullptr) {
        fprintf(stderr, "ERROR: SDL_CreateWindow failed");
        SDL_Log("%s", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    // Get window surface
    SDL_Surface* s = SDL_GetWindowSurface(win);

    // Display error and exit if surface fetch failed
    if (s == nullptr) {
        fprintf(stderr, "ERROR: SDL_GetWindowSurface failed");
        SDL_Log("%s", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    SDL_SetSurfaceBlendMode(s, SDL_BLENDMODE_BLEND);

    // END SDL INITIALIZATION STUFF ///

    #define RGB(r,g,b) (SDL_MapSurfaceRGB(s,r,g,b))

    drawSquare(s,{0,0},{WINDOW_W,WINDOW_H}, RGB(10,10,10));


    GenePool_s pool(2,3,1,DEFAULT_PARAMETERS,nullptr);

    Genome_s genome(pool,true);
    {
        for (int i = 0; i < 10; ++i)
            genome.mutateAddNode();
        for (int i = 0; i < 4; ++i)
            genome.mutateAddConnection();
    }

    std::vector<glm::vec2> positions;
    arrangeGraph(genome, positions, 4096);
    drawGraph(s, genome, positions);

    
    // Main loop
    SDL_Event e;
    while(SDL_PollEvent(&e) == false || e.type != SDL_EVENT_QUIT) {

        // "tickle the surface"
        SDL_UpdateWindowSurface(win);

        // Sort-of cap FPS
        SDL_DelayNS(frametime_ns);
    }
}