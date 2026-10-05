#include "../inc/GraphVis.hpp"
#include "../inc/randutil.hpp"
#include "NEAT.hpp"
#include <SDL3/SDL_surface.h>
#include <cmath>
#include <glm/detail/qualifier.hpp>
#include <glm/geometric.hpp>

int glm::doti(const vec2i a, const vec2i b) {
    return a.x*b.x + a.y*b.y;
}

void arrangeGraph(const Genome_s& genome, std::vector<glm::vec2>& positions, const size_t iterations) {
    typedef std::pair<NodeID_t,NodeID_t> conn_t;

    // Initialize all positions
    positions.clear();
    positions.reserve(genome.getNodeCount());
    for (NodeID_t i = 0; i < genome.getNodeCount(); ++i)
        positions.emplace_back(0,0);

    // Add input positions
    const float_t INPUT_SPACING = 1.0 / (float_t)(genome.getInputNodeCount() + (genome.hasBias() ? 1 : 0));
    for (NodeID_t i = genome.getInputNodeStart(); i < genome.getInputNodeEnd(); ++i) {
        positions[i] = {
            0, 
            INPUT_SPACING * ((float_t)(i-genome.getInputNodeStart()) + 0.5)
        };
    }
    // Add bias
    if (genome.hasBias()) {
        positions[genome.getBiasNode()] = {0, 1.0 - (0.5*INPUT_SPACING) };
    }

    // Add outputs
    const float_t OUTPUT_SPACING = 1.0 / (float_t)(genome.getOutputNodeCount());
    for (NodeID_t i = genome.getOutputNodeStart(); i < genome.getOutputNodeEnd(); ++i) {
        positions[i] = {
            1, 
            OUTPUT_SPACING * ((float_t)(i-genome.getOutputNodeCount()) + 0.5)
        };
    }

    // Add hidden nodes in random positons
    constexpr float_t HIDDEN_MARGIN = 0.2; // [0.0, 0.5)
    #define RAND_IN_MARGIN() (RandUtil::randF()*(1-(2*HIDDEN_MARGIN)) + HIDDEN_MARGIN)
    for (NodeID_t i = genome.getHiddenNodeStart(); i < genome.getHiddenNodeEnd(); ++i) {
        positions.at(i) = {
            RAND_IN_MARGIN(),
            RAND_IN_MARGIN(),
        };
    }


    std::vector<conn_t> connections;
    for (const Gene_s& gene : genome.genome) {
        if (gene.enabled == false) continue;
        if (gene.weight == 0) continue;

        connections.emplace_back(gene.FROM, gene.TO);
    }

    for (size_t i = 0; i < iterations; ++i) {

        const float_t NODE_SPACING = 1.0 / std::sqrt(genome.getNodeCount());

        constexpr float_t PERTURB = 0.08;
        constexpr float_t NODE_REPEL_AMOUNT = 0.5;

        const NodeID_t CHECK_COUNT = (NodeID_t)(std::sqrt(genome.getNodeCount())) + 5;
        for (NodeID_t n = genome.getHiddenNodeStart(); n < genome.getHiddenNodeEnd(); ++n) {
            glm::vec2& n_pos = positions[n];

            for (NodeID_t o = 0; o < genome.getNodeCount(); ++o) {
                const glm::vec2& o_pos = positions[o];

                // Direction from `n` to `o`
                const glm::vec2 diff = o_pos - n_pos;
                const float_t dist = glm::length(diff);

                // Ignore if distance is too close
                if (dist < 1e-3) {
                    n_pos.x += (RandUtil::randF()-0.5)*PERTURB;
                    n_pos.y += (RandUtil::randF()-0.5)*PERTURB;
                    continue;
                }
                // Ignore if distance is too great
                if (dist >= NODE_SPACING) {
                    continue;
                }

                const float_t scale = NODE_REPEL_AMOUNT * std::max(1.0/dist - 1.0/NODE_SPACING, 0.0) / (float_t)genome.getNodeCount();

                n_pos -= scale * glm::normalize(diff);
            }
        }

        constexpr float_t CONNECTION_PULL_AMOUNT = 0.5;
        for (const conn_t& conn : connections) {
            glm::vec2& f_pos = positions[conn.first];
            glm::vec2& t_pos = positions[conn.second];

            // Direction from `f` to `t`
            const glm::vec2 diff = t_pos - f_pos;
            const float_t dist = glm::length(diff);

            const float_t scale = std::max(float_t{0}, dist-NODE_SPACING);

            const bool f_hidden = genome.isHiddenNode(conn.first);
            const bool t_hidden = genome.isHiddenNode(conn.second);

            if (f_hidden && !t_hidden) {
                f_pos += diff * CONNECTION_PULL_AMOUNT * scale;
            }
            else if (t_hidden && !f_hidden) {
                t_pos -= diff * CONNECTION_PULL_AMOUNT * scale;
            }
            else if (f_hidden && t_hidden) {
                f_pos += diff * CONNECTION_PULL_AMOUNT * scale * 0.5f;
                t_pos -= diff * CONNECTION_PULL_AMOUNT * scale * 0.5f;
            }
        }

        constexpr float_t CENTER_TREND = 0.08;
        const float_t CENTER_THRESH = 0.5 - NODE_SPACING;

        for (NodeID_t n = genome.getHiddenNodeStart(); n < genome.getHiddenNodeEnd(); ++n) {
            glm::vec2& pos = positions[n];

            pos.x = std::min(float_t{1}, std::max(float_t{0},pos.x));
            pos.y = std::min(float_t{1}, std::max(float_t{0},pos.y));

            glm::vec2 cd = glm::vec2{0.5} - pos;

            if (std::abs(cd.x) > CENTER_THRESH) 
                cd.x -= std::signbit(cd.x)*CENTER_THRESH;
            if (std::abs(cd.y) > CENTER_THRESH) 
                cd.y -= std::signbit(cd.y)*CENTER_THRESH;

            pos += CENTER_TREND * cd;
        }
    }
}

inline size_t index(const SDL_Surface* img, const glm::vec2i pos) {
    return pos.x + pos.y*img->w;
}

inline void drawTo(SDL_Surface* img, const Uint32 color, const glm::vec2i pos) {
    const size_t i = index(img,pos);
    if (i > img->h*img->w) return;
    ((Uint32*)(img->pixels))[i] = color;
}

void drawSquare(SDL_Surface* img, const glm::vec2i lower, const glm::vec2i upper, const Uint32 color) {
    for (
        glm::vec2i pos = {lower.x, lower.y};
        pos.y <= upper.y;
        (++pos.x > upper.x) && (pos.x = lower.x, ++pos.y)
    ) {
        drawTo(img, color, pos);
    }
}

void drawCircle(SDL_Surface* img, const glm::vec2i center, const int rad, const Uint32 color) {
    glm::vec2i rel;
    const int r2 = rad*rad;
    for (
        glm::vec2i pos = {center.x-rad, center.y-rad};
        pos.y <= center.y+rad;
        (++pos.x > center.x+rad) && (pos.x = center.x-rad, ++pos.y)
    ) {
        rel = pos - center;
        if (rel.x*rel.x + rel.y*rel.y > r2) continue;
        drawTo(img, color, pos);
    }
}

void drawLine(SDL_Surface* img, const glm::vec2i p0, const glm::vec2i p1, const Uint32 color, const int thick) {
    const bool X_POSI = p0.x < p1.x;
    const bool Y_POSI = p0.y < p1.y;
    #define STEP_X(p) { if (X_POSI) ++p.x; else --p.x; }
    #define STEP_Y(p) { if (Y_POSI) ++p.y; else --p.y; }
    if (thick < 1) return;

    glm::vec2i D = p1-p0;
    const glm::vec2i L{-D.y, D.x};
    const bool X_FIRST = std::abs(D.x) > std::abs(D.y);

    glm::vec2i pos = p0;
    int res = 0;

    drawTo(img, color, pos);

    if (X_FIRST) {
        const int THRESH = std::abs(L.y) / 2;

        while (pos.x != p1.x) {
            // Step
            STEP_X(pos);

            // Line equation
            res = glm::doti(pos - p0, L);

            // Step in Y if drifting away from line
            if (std::abs(res) > THRESH) {
                STEP_Y(pos);
            }

            for (int y = pos.y - thick/2; y < pos.y + (thick+1)/2; ++y) {
                drawTo(img, color, {pos.x, y});
            }
        }
    }
    else {
        const int THRESH = std::abs(L.x) / 2;

        while (pos.y != p1.y) {
            // Step
            STEP_Y(pos);

            // Line equation
            res = glm::doti(pos - p0, L);

            // Step in Y if drifting away from line
            if (std::abs(res) > THRESH) {
                STEP_X(pos);
            }

            for (int x = pos.x - thick/2; x < pos.x + (thick+1)/2; ++x) {
                drawTo(img, color, {x, pos.y});
            }
        }
    }
}

void drawTriangle(SDL_Surface* img, const glm::vec2i A, const glm::vec2i B, const glm::vec2i C, const Uint32 color) {
    // Create AB and AC vectors
    const glm::vec2i
        AB = B-A,
        BC = C-B,
        CA = A-C;

    // Define vectors used for line equations
    const glm::vec2i 
        L_AB = {AB.y,-AB.x},
        L_BC = {BC.y,-BC.x},
        L_CA = {CA.y,-CA.x};
    
    const int 
        T_AB = std::abs(glm::doti(L_AB, {1,1})) / 4,
        T_BC = std::abs(glm::doti(L_BC, {1,1})) / 4,
        T_CA = std::abs(glm::doti(L_CA, {1,1})) / 4;

    // Define AABB
    const glm::vec2i
        AABB_l = {
            std::min(std::min(A.x, B.x), C.x),
            std::min(std::min(A.y, B.y), C.y),
        }, 
        AABB_h = {
            std::max(std::max(A.x, B.x), C.x),
            std::max(std::max(A.y, B.y), C.y),
        };

    const bool CCW = (glm::doti(L_AB, C-A) < 0);
    
    // For each pixel in the AABB
    for (
        glm::vec2i pos = {AABB_l.x, AABB_l.y};
        pos.y <= AABB_h.y;
        (++pos.x > AABB_h.x) && (pos.x = AABB_l.x, ++pos.y)
    ) {
        // Check if outside the traingle according to any single line
        // Use CCW/CW to determine if "outside" means positive or negative result from line eq
        if (!CCW) {
            if (glm::doti(L_AB, pos-A) < T_AB) continue;
            if (glm::doti(L_BC, pos-B) < T_BC) continue;
            if (glm::doti(L_CA, pos-C) < T_CA) continue;
        }
        else {
            if (glm::doti(L_AB, pos-A) > T_AB) continue;
            if (glm::doti(L_BC, pos-B) > T_BC) continue;
            if (glm::doti(L_CA, pos-C) > T_CA) continue;
        }
        drawTo(img, color, pos);
    }
}

void drawArrow(SDL_Surface* img, const glm::vec2i start, const glm::vec2i end, const int line_width, const int arrow_size, const Uint32 color) {
    const glm::vec2i dir = end-start;
    const glm::vec2i orth = {dir.y, -dir.x};

    const glm::vec2i center_off = (dir*arrow_size*4) / (glm::lengthi(dir)*5);
    const glm::vec2i side_off_l = (dir*arrow_size) / (glm::lengthi(dir));
    const glm::vec2i side_off_s = (orth*(arrow_size+1)*3) / (glm::lengthi(orth)*5);

    const glm::vec2i A = end - center_off;
    const glm::vec2i B = end - side_off_l + side_off_s;
    const glm::vec2i C = end - side_off_l - side_off_s;

    const glm::vec2i center = (3*A + end) / 4;

    drawLine(img, start, center, color, line_width);

    drawTriangle(img, A,B,end, color);
    drawTriangle(img, A,C,end, color);
}

void drawGraph(SDL_Surface* img, const Genome_s& genome, std::vector<glm::vec2>& positions) {
    typedef struct { Uint8 r, g, b, a=255; } color_t;

    constexpr size_t COLOR_COUNT = 12;
    constexpr color_t CONNECTION_COLORS[COLOR_COUNT] = {
        {255,0,0}, // Red
        {0,255,0}, // Green
        {0,0,255}, // Blue
        {0,240,240}, // Teal
        {240,0,240}, // Purple
        {240,240,0}, // Yellow
        {0,250,120}, // Green-teal
        {120,0,250}, // Indigo
        {250,120,0}, // Orange
        {0,120,250}, // Blue-teal
        {250,0,120}, // Magenta
        {120,250,0}, // Green-yellow
    };
    constexpr color_t NODE_COLOR = {220,255,240,255};
    constexpr color_t NODE_COLOR_OVERLAY = {NODE_COLOR.r, NODE_COLOR.g, NODE_COLOR.b,127};
    #define COLOR_TO_INT(color) (SDL_MapSurfaceRGBA(img, color.r, color.g, color.b, color.a))

    const Uint32 NODE_ICOLOR = COLOR_TO_INT(NODE_COLOR);
    const Uint32 NODE_ICOLOR_OVERLAY = COLOR_TO_INT(NODE_COLOR_OVERLAY);

    constexpr float_t MARG = 0.05;
    constexpr float_t NMARG = 1.0 - 2*MARG;
    const float_t SCALE = (1.0 / std::sqrt(genome.getNodeCount())) * glm::lengthi({img->h, img->w}) * NMARG;

    const int LINE_WIDTH = std::max(1.0, SCALE/80.0);
    const int ARROW_SIZE = std::max(4.0, SCALE/24.0);
    const int NODE_SIZE = std::max(4.0, 5 + SCALE/24.0);

    #define MAP_POS(fpos) (glm::vec2i{(int)((fpos.x*NMARG + MARG)*img->w), (int)((fpos.y*NMARG + MARG)*img->h)})

    for (const auto& node_pos : positions) {
        drawCircle(img, MAP_POS(node_pos), NODE_SIZE, NODE_ICOLOR);
    }

    size_t i = 0;
    for (const auto& gene : genome.genome) {
        if (!gene.enabled) continue;
        if (std::abs(gene.weight) < 1e-5) continue;

        glm::vec2i spos = MAP_POS(positions[gene.FROM]);
        glm::vec2i epos = MAP_POS(positions[gene.TO]);
        const glm::vec2i diff = epos - spos;

        // If nodes are too close, don't draw arrow
        if (std::abs(diff.x) < 2*NODE_SIZE && std::abs(diff.y) < 2*NODE_SIZE)
            continue;

        spos += (diff*NODE_SIZE) / glm::lengthi(diff);
        epos -= (diff*NODE_SIZE) / glm::lengthi(diff);


        const color_t& color = CONNECTION_COLORS[i];
        const Uint32 icolor = COLOR_TO_INT(color);
        (++i > COLOR_COUNT && (i = 0));

        drawArrow(img, spos, epos, LINE_WIDTH, ARROW_SIZE, icolor);
    }

    for (const auto& node_pos : positions) {
        drawCircle(img, MAP_POS(node_pos), NODE_SIZE, NODE_ICOLOR_OVERLAY);
    }

    #undef COLOR_TO_INT
    #undef RAND_AMNT
    #undef RANDOMIZE_COLOR
}