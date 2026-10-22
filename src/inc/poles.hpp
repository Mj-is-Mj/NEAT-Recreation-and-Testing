#pragma once

#include <vector>

/*

    System Parameters:
        g:      gravity
        m_c:    mass of cart
        m_pn:   mass of pole n
        l_n:    half-length of pole n
        h:      time step
        u_c:    coefficient of friction of the cart on the track
        u_pi:   coefficient of friction for the ith hinge

    System state:
        x:      cart position
        dx:     cart velocity
        th_n:   angle of nth pole
        dth_n:  angular velocity of nth pole

    Other terms/whatnot:
        t:      time
        F:      force

    RK4:
        - Evaluates the rate of change of a part of the system state 
        four separate times per step
        - Take: 
            - `y(t)` as the approximate system state at a given time `t`
            - `f(y_0,t_0)` as the rate of change of the system state given the 
            current system state `y_0` and time `t_0`. I.e., for a simple function
            `g(x) = ...`, `f(g(t),t) = g'(t)`
            - `h` as the time step (i.e. `{new time} = {current time} + h`)
        - Then:
            - `k_1 = f(y(t),  t)`
            - `k_2 = f(y(t) + k_1*h/2,  t + h/2)`
            - `k_3 = f(y(t) + k_2*h/2,  t + h/2)`
            - `k_4 = f(y(t) + k_3*h,  t + h)`
            - `m = (k_1 + 2*(k_2+k_3) + k_4) / 6`
            - `y(t+h) = y(t) + mh`

    System state rates of change:
        // According to "Evolving Neural Network Controllers for Unstable Systems"
        // by Alexis P. Wieland

        // Effective force of pole i on the cart
        F_i = m_pi*l_i*(dth_i^2)*sin(th_i) + (3/4)*m_pi*cos(th_i)*((u_pi*dth_i)/(m_pi*l_i) + g*sin(th_i))
        
        // Effective mass of pole i
        m_epi = m_i*(1 - (3/4)*cos^2(th_i))

        // Accelleration of the cart
        ddx = (F - u_c*sgn(dx) + sum(i=1,N)(F_i)) / (m_c + sum(i=1,N)(m_epi))

        // Angular accelleration of pole i
        ddth_i = (-3/(4*l_i))*(ddx*cos(th_i) + g*sin(th_i) + (u_pi*dth_i)/(m_pi*l_i))
    
    Other notes:
        When implementing pole-balancing with RK4, we don't use the `t`/"time" parameter
        of the rate of change function `f`, as the formulas modeling the physics of the 
        problem act only on the given state to calculate acelleration, and do not use
        the timestamp of that state. 

*/
namespace PoleProblems {
    typedef float Float_t;

    struct Pole_s {
        const Float_t mass;
        const Float_t length; // **This is technically the half-length, will adjust formulas to make it the actual length
        const Float_t hinge_friction;
        const Float_t initial_angle;
        const Float_t initial_velocity;
    };

    struct Cart_s {
        const Float_t mass;
        const Float_t track_range;
        const Float_t initial_position;
        const Float_t initial_velocity;
        const Float_t track_friction;
    };


    struct CartWithPoles_s {
        struct Parameters_s {
            Float_t step_size;
            Float_t gravity;
        };

        struct State_s {
            struct PoleState_s {
                Float_t theta;
                Float_t dtheta;

                PoleState_s(const Float_t theta, const Float_t dtheta)
                    : theta(theta)
                    , dtheta(dtheta)
                {}

                PoleState_s(const Pole_s& pole)
                    : theta(pole.initial_angle)
                    , dtheta(pole.initial_velocity)
                {}

                PoleState_s(const Pole_s&& pole)
                    : theta(pole.initial_angle)
                    , dtheta(pole.initial_velocity)
                {}
            };
            struct CartState_s {
                Float_t x;
                Float_t dx;

                CartState_s(const Float_t x, const Float_t dx)
                    : x(x)
                    , dx(dx)
                {}

                CartState_s(const Cart_s& cart)
                    : x(cart.initial_position)
                    , dx(cart.initial_velocity)
                {}
            };

            CartState_s cart;
            std::vector<PoleState_s> poles;

            template<typename... Poles_tmp>
            State_s(
                const Cart_s& cart,
                const Poles_tmp... poles
            )
                : cart(cart)
                , poles{poles...}
            {}

            State_s(
                const Cart_s& cart,
                std::vector<PoleState_s> poles
            )
                : cart(cart)
                , poles{poles}
            {}

            State_s(const State_s& other)
                : cart(other.cart)
                , poles(other.poles)
            {}

            State_s(const CartState_s& other_cart, const std::vector<PoleState_s> other_poles)
                : cart(other_cart)
                , poles(other_poles)
            {}

            State_s operator+(const State_s& other) const;
            State_s operator*(const Float_t scale) const;
            inline State_s& operator+=(const State_s& other) { *this = *this + other; return *this; }
        };

        private:
            const size_t _pole_count;
            const Cart_s _cart;
            const std::vector<Pole_s> _poles;
            State_s _state;

            State_s _ROC(const Float_t cart_force, const State_s current_state) const;

        public:
            const Parameters_s PARAMTERS;

            template<typename... Poles_tmp>
            CartWithPoles_s(
                const Parameters_s params,
                const Cart_s cart,
                const Poles_tmp... poles
            ) 
                : _pole_count(sizeof...(poles))
                , _cart(cart)
                , _poles{poles...}
                , _state(cart, poles...)
                , PARAMTERS(params)
            {}

            constexpr const State_s& getState() { return _state; }

            void step(const Float_t cart_force, const Float_t step_size);
            inline void step(const Float_t cart_force) { step(cart_force, PARAMTERS.step_size); }


    };


}