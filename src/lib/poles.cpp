#include "../inc/poles.hpp"
#include "mathutil.hpp"

#include <cassert>
#include <cmath>

namespace PoleProblems {


CartWithPoles_s::State_s CartWithPoles_s::State_s::operator+(const CartWithPoles_s::State_s& other) const {
    assert(other.poles.size() == this->poles.size());

    // Copy this state
    State_s new_state(*this);

    // Add cart data
    new_state.cart.x += other.cart.x;
    new_state.cart.dx += other.cart.dx;

    // Add pole data
    for (size_t i = 0; i < poles.size(); ++i) {
        new_state.poles[i].theta += other.poles[i].theta;
        new_state.poles[i].dtheta += other.poles[i].dtheta;
    }

    return new_state;
}

CartWithPoles_s::State_s CartWithPoles_s::State_s::operator*(const Float_t scale) const {
    // Copy this state
    State_s new_state(*this);

    // Add cart data
    new_state.cart.x *= scale;
    new_state.cart.dx *= scale;

    // Add pole data
    for (size_t i = 0; i < poles.size(); ++i) {
        new_state.poles[i].theta *= scale;
        new_state.poles[i].dtheta *= scale;
    }

    return new_state;
}

CartWithPoles_s::State_s CartWithPoles_s::_ROC(const Float_t cart_force, const State_s current_state) const {
    Float_t effective_pole_force = 0;
    Float_t effective_pole_mass = 0;

    // Compute effective forces and masses
    for (size_t i = 0; i < _pole_count; ++i) {
        const Pole_s& pole = _poles[i];
        const State_s::PoleState_s& pole_state = current_state.poles[i];

        effective_pole_force += (
            (pole.mass * pole.length * pole_state.dtheta*pole_state.dtheta * Math::sin(pole_state.theta))
            + (0.75*pole.mass*Math::cos(pole_state.theta))
            * (
                (pole.hinge_friction*pole_state.dtheta) / (pole.mass*pole.length)
                + PARAMTERS.gravity * sin(pole_state.theta)
            )
        );

        effective_pole_mass += (
            pole.mass*(1.0 - 0.75 * Math::cos2(pole_state.theta))
        );
    }

    /*
        Note: Since this yeilds the rate of change, Nth derivatives will be
        stored in (N-1)th derivatives. E.g., cart position will hold the computed cart velocity
    */

    // Cart positional acceleration
    const Float_t ddx = (
        (cart_force - _cart.track_friction*Math::sgn(current_state.cart.dx) + effective_pole_force)
        / (_cart.mass + effective_pole_mass)
    );

    // Pole angular accelerations
    std::vector<State_s::PoleState_s> pole_changes; pole_changes.reserve(_pole_count);

    for (size_t i = 0 ; i < _pole_count; ++i) {
        const Pole_s& pole = _poles[i];
        const State_s::PoleState_s& pole_state = current_state.poles[i];

        // Add a new pole state
        pole_changes.push_back({0,0});
        State_s::PoleState_s& change = pole_changes.back();

        // Update derivative
        change.theta = pole_state.dtheta;

        // Compute second derivative
        change.dtheta = ((Float_t{-0.75} / pole.length) * (
              (ddx*Math::cos(pole_state.theta))
            + (PARAMTERS.gravity*Math::sin(pole_state.theta))
            + (pole.hinge_friction*pole_state.dtheta) / (pole.mass*pole.length)
        ));
    }

    return {
        State_s::CartState_s{current_state.cart.dx, ddx},
        std::move(pole_changes)
    };
}

void CartWithPoles_s::step(const Float_t cart_force, const Float_t step_size) {
    const State_s k1 = _ROC(cart_force, _state);
    const State_s k2 = _ROC(cart_force, _state + k1*(step_size*0.5f));
    const State_s k3 = _ROC(cart_force, _state + k2*(step_size*0.5f));
    const State_s k4 = _ROC(cart_force, _state + k3*step_size);

    const State_s mh = (k1 + (k2+k3)*2.f + k4) * (step_size / 6.f);
    
    _state = _state + mh;
};


}