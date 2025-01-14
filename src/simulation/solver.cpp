#include "solver.h"

#include <Eigen/Core>

using Eigen::Vector3f;

// External Force does not changed.

// Function to calculate the derivative of KineticState
KineticState derivative(const KineticState& state)
{
    return KineticState(state.velocity, state.acceleration, Eigen::Vector3f(0, 0, 0));
}

// Function to perform a single Forward Euler step
KineticState forward_euler_step([[maybe_unused]] const KineticState& previous,
                                const KineticState& current)
{
    KineticState next;
    next.position     = current.position + time_step * current.velocity;
    next.velocity     = current.velocity + time_step * current.acceleration;
    next.acceleration = current.acceleration;
    return next;
}

// Function to perform a single Runge-Kutta step
KineticState runge_kutta_step([[maybe_unused]] const KineticState& previous,
                              const KineticState& current)
{
    KineticState next;
    Vector3f k1_a, k2_a, k3_a, k4_a;
    Vector3f k1_v, k2_v, k3_v, k4_v;
    k1_a = current.acceleration;
    k2_a = current.acceleration;
    k3_a = current.acceleration;
    k4_a = current.acceleration;
    next.velocity = current.velocity + time_step * (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6;
    k1_v          = current.velocity;
    k2_v          = k1_v + time_step * k1_a / 2;
    k3_v          = k2_v + time_step * k2_a / 2;
    k4_v          = current.velocity + time_step * current.acceleration;
    next.position = current.position + time_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;
    next.acceleration = current.acceleration;
    return next;
}

// Function to perform a single Backward Euler step
KineticState backward_euler_step([[maybe_unused]] const KineticState& previous,
                                 const KineticState& current)
{
    KineticState next;
    next.acceleration = current.acceleration;
    next.velocity     = current.velocity + time_step * next.acceleration;
    next.position     = current.position + time_step * next.velocity;
    return next;
}

// Function to perform a single Symplectic Euler step
KineticState symplectic_euler_step(const KineticState& previous, const KineticState& current)
{
    KineticState next;
    (void) previous;
    next.velocity     = current.velocity + time_step * current.acceleration;
    next.position     = current.position + time_step * next.velocity;
    next.acceleration = current.acceleration;
    return next;
}
