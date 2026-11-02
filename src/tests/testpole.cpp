#include "../inc/poles.hpp"

#include <iostream>


void operator<<(std::ostream& out, const PoleProblems::CartWithPoles_s::State_s& state) {
    out << "Cart: " << std::endl
        << "\tPos = " << state.cart.x << std::endl
        << "\tVel = " << state.cart.dx << std::endl;
    
    for (size_t i = 0; i < state.poles.size(); ++i) {
        out << "Pole " << i << ": " << std::endl
            << "\tAng = " << state.poles[i].theta << std::endl
            << "\tVel = " << state.poles[i].dtheta << std::endl;
    }
}

void csv(std::ostream& out, const PoleProblems::CartWithPoles_s::State_s& state) {
    out << std::fixed << state.cart.x << ", "
        << std::fixed << state.cart.dx << ", ";
    
    for (size_t i = 0; i < state.poles.size(); ++i) {
        out << std::fixed << state.poles[i].theta << ", "
            << std::fixed << state.poles[i].dtheta << ", ";
    }
    out << std::endl;
}

// See result in desmos - https://www.desmos.com/calculator/vtyy0brmun
int main() {
    PoleProblems::CartWithPoles_s pole_balance{};

    for (size_t i = 0; i < 256; ++i) {
        // std::cout << "----- STEP " << i << " -----" << std::endl;
        // std::cout << pole_balance.getState();
        csv(std::cout, pole_balance.getState());
        pole_balance.step(0);
    };

    csv(std::cout, pole_balance.getState());
    // std::cout << "----- FINAL -----" << std::endl;
    // std::cout << pole_balance.getState();
}