"""
Jack Weinheimer
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sin, cos, exp, pi, e


# =============================================================================
# Helper Functions
# =============================================================================

def true_percent_relative_error(true_val, approx_val):
    """
    Calculate true percent relative error.
    εT (%) = 100 × |true - approx| / |true|
    """
    return 100 * abs((true_val - approx_val) / true_val)


# =============================================================================
# Problem 4.12: Taylor Series Expansions (0th through 3rd order)
# =============================================================================

def f(x):
    """Function: f(x) = 25x³ - 6x² + 7x - 88"""
    return 25 * x**3 - 6 * x**2 + 7 * x - 88


def f_prime(x):
    """First derivative: f'(x) = 75x² - 12x + 7"""
    return 75 * x**2 - 12 * x + 7


def f_double_prime(x):
    """Second derivative: f''(x) = 150x - 12"""
    return 150 * x - 12


def f_triple_prime(x):
    """Third derivative: f'''(x) = 150"""
    return 150


def taylor_expansion(x, x0, order):
    """
    Compute Taylor series expansion of order n about x0, evaluated at x.
    Tn(x) = Σ(k=0 to n) [f^(k)(x0) / k!] * (x - x0)^k
    """
    derivatives = [f(x0), f_prime(x0), f_double_prime(x0), f_triple_prime(x0)]

    result = 0
    for k in range(order + 1):
        result += (derivatives[k] / factorial(k)) * (x - x0)**k

    return result


def problem_4_12():
    """
    Problem 4.12: Use zeroth- through third-order Taylor series expansions
    to predict f(3) using base point x0 = 1.
    """
    print("=" * 80)
    print("Problem 4.12: Taylor Series Expansions")
    print("=" * 80)
    print(f"Function: f(x) = 25x³ - 6x² + 7x - 88")
    print(f"Base point: x0 = 1")
    print(f"Evaluation point: x = 3")
    print()

    x0 = 1
    x = 3
    true_value = f(x)

    print(f"True value f(3) = {true_value:.10f}")
    print()

    # Create arrays for plotting
    orders = [0, 1, 2, 3]
    approximations = []
    errors = []

    for order in orders:
        approx = taylor_expansion(x, x0, order)
        error = true_percent_relative_error(true_value, approx)
        approximations.append(approx)
        errors.append(error)

        print(f"Order {order}:")
        print(f"  Approximation: {approx:.10f}")
        print(f"  True % Relative Error: {error:.10f}%")
        print()

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot approximations vs true value
    ax1.plot(orders, [true_value] * len(orders), 'r--', label='True Value', linewidth=2)
    ax1.plot(orders, approximations, 'bo-', label='Taylor Approximations', markersize=8)
    ax1.set_xlabel('Taylor Series Order', fontsize=12)
    ax1.set_ylabel('f(3)', fontsize=12)
    ax1.set_title('Taylor Series Approximations vs True Value', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)

    # Plot errors
    ax2.semilogy(orders, errors, 'ro-', markersize=8, linewidth=2)
    ax2.set_xlabel('Taylor Series Order', fontsize=12)
    ax2.set_ylabel('True % Relative Error (%)', fontsize=12)
    ax2.set_title('Error Reduction with Taylor Series Order', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)

    plt.tight_layout()
    plt.savefig('problem_4_12_taylor_series.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'problem_4_12_taylor_series.png'")
    plt.close()

    return approximations, errors


# =============================================================================
# Problem 4.15: First Derivative Approximations
# =============================================================================

def forward_difference(func, x, h):
    """Forward difference: [f(x+h) - f(x)] / h"""
    return (func(x + h) - func(x)) / h


def backward_difference(func, x, h):
    """Backward difference: [f(x) - f(x-h)] / h"""
    return (func(x) - func(x - h)) / h


def centered_difference(func, x, h):
    """Centered difference: [f(x+h) - f(x-h)] / (2h)"""
    return (func(x + h) - func(x - h)) / (2 * h)


def problem_4_15():
    """
    Problem 4.15: Estimate first derivative at x=2 using forward, backward,
    and centered differences with h=0.25.
    """
    print("\n" + "=" * 80)
    print("Problem 4.15: First Derivative Approximations")
    print("=" * 80)
    print(f"Function: f(x) = 25x³ - 6x² + 7x - 88")
    print(f"Point: x = 2")
    print(f"Step size: h = 0.25")
    print()

    x = 2
    h = 0.25
    true_derivative = f_prime(x)

    fwd = forward_difference(f, x, h)
    bwd = backward_difference(f, x, h)
    ctr = centered_difference(f, x, h)

    print(f"True derivative f'(2) = {true_derivative:.10f}")
    print()

    print(f"Forward Difference (O(h)):")
    print(f"  Approximation: {fwd:.10f}")
    print(f"  Absolute Error: {abs(true_derivative - fwd):.10f}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_derivative, fwd):.10f}%")
    print()

    print(f"Backward Difference (O(h)):")
    print(f"  Approximation: {bwd:.10f}")
    print(f"  Absolute Error: {abs(true_derivative - bwd):.10f}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_derivative, bwd):.10f}%")
    print()

    print(f"Centered Difference (O(h²)):")
    print(f"  Approximation: {ctr:.10f}")
    print(f"  Absolute Error: {abs(true_derivative - ctr):.10f}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_derivative, ctr):.10f}%")
    print()

    print("Interpretation:")
    print("The centered difference is more accurate due to its O(h²) truncation error,")
    print("while forward and backward differences have O(h) truncation error.")
    print("The centered difference cancels out the first-order error terms in the")
    print("Taylor series expansion, leading to higher accuracy.")


# =============================================================================
# Problem 4.16: Second Derivative Approximations
# =============================================================================

def second_derivative_centered(func, x, h):
    """
    Centered difference for second derivative:
    [f(x+h) - 2f(x) + f(x-h)] / h²
    """
    return (func(x + h) - 2 * func(x) + func(x - h)) / h**2


def problem_4_16():
    """
    Problem 4.16: Estimate second derivative at x=2 using centered difference
    with h=0.2 and h=0.1.
    """
    print("\n" + "=" * 80)
    print("Problem 4.16: Second Derivative Approximations")
    print("=" * 80)
    print(f"Function: f(x) = 25x³ - 6x² + 7x - 88")
    print(f"Point: x = 2")
    print()

    x = 2
    true_second_derivative = f_double_prime(x)

    print(f"True second derivative f''(2) = {true_second_derivative:.10f}")
    print()

    for h in [0.2, 0.1]:
        approx = second_derivative_centered(f, x, h)
        error = abs(true_second_derivative - approx)
        rel_error = true_percent_relative_error(true_second_derivative, approx)

        print(f"Step size h = {h}:")
        print(f"  Approximation: {approx:.10f}")
        print(f"  Absolute Error: {error:.10f}")
        print(f"  True % Relative Error: {rel_error:.10f}%")
        print()

    print("Interpretation:")
    print("As h decreases from 0.2 to 0.1, the error decreases by approximately a factor of 4,")
    print("which is consistent with the O(h²) truncation error of the centered difference formula.")
    print("This demonstrates the quadratic convergence of the method.")


# =============================================================================
# Problem 4.18: Taylor Series with Error Constraint
# =============================================================================

def problem_4_18():
    """
    Problem 4.18: Determine highest-order Taylor series expansion with
    maximum error of 0.015 on [0, π].
    Function: f(x) = x - 1 - 0.5·sin(x)
    Base point: a = π/2
    """
    print("\n" + "=" * 80)
    print("Problem 4.18: Taylor Series with Error Constraint")
    print("=" * 80)
    print(f"Function: f(x) = x - 1 - 0.5·sin(x)")
    print(f"Base point: a = π/2")
    print(f"Interval: [0, π]")
    print(f"Maximum allowed error: 0.015")
    print()

    def g(x):
        """g(x) = x - 1 - 0.5·sin(x)"""
        return x - 1 - 0.5 * np.sin(x)

    # Derivatives at a = π/2
    a = np.pi / 2

    # g(π/2) = π/2 - 1 - 0.5·sin(π/2) = π/2 - 1 - 0.5
    # g'(x) = 1 - 0.5·cos(x), g'(π/2) = 1 - 0
    # g''(x) = 0.5·sin(x), g''(π/2) = 0.5
    # g'''(x) = 0.5·cos(x), g'''(π/2) = 0
    # g''''(x) = -0.5·sin(x), g''''(π/2) = -0.5

    derivs_at_a = [
        a - 1 - 0.5,           # g(a)
        1.0,                    # g'(a)
        0.5,                    # g''(a)
        0.0,                    # g'''(a)
        -0.5,                   # g''''(a)
        0.0,                    # g'''''(a)
        0.5,                    # g''''''(a)
    ]

    x_vals = np.linspace(0, np.pi, 1000)
    true_vals = g(x_vals)
    max_error_threshold = 0.015

    max_order_found = 0

    for order in range(1, 10):
        # Compute Taylor series of given order
        taylor_vals = np.zeros_like(x_vals)
        for k in range(order + 1):
            if k < len(derivs_at_a):
                taylor_vals += (derivs_at_a[k] / factorial(k)) * (x_vals - a)**k

        # Compute maximum error on the interval
        errors = np.abs(true_vals - taylor_vals)
        max_error = np.max(errors)

        print(f"Order {order}: Maximum error = {max_error:.10f}")

        if max_error <= max_error_threshold:
            max_order_found = order
            print(f"  ✓ Satisfies error constraint (≤ {max_error_threshold})")
        else:
            print(f"  ✗ Exceeds error constraint")

    print()
    print(f"Highest-order Taylor series with error ≤ {max_error_threshold}: Order {max_order_found}")


# =============================================================================
# Problem 4.19: Graphical Comparison of Finite Differences
# =============================================================================

def problem_4_19():
    """
    Problem 4.19: Graph finite difference approximations for first and second
    derivatives along with theoretical values.
    Function: f(x) = x³ - 2x + 4
    Interval: [-2, 2], h = 0.25
    """
    print("\n" + "=" * 80)
    print("Problem 4.19: Graphical Comparison of Finite Differences")
    print("=" * 80)
    print(f"Function: f(x) = x³ - 2x + 4")
    print(f"Interval: [-2, 2]")
    print(f"Step size: h = 0.25")
    print()

    def h_func(x):
        """h(x) = x³ - 2x + 4"""
        return x**3 - 2*x + 4

    def h_prime_exact(x):
        """h'(x) = 3x² - 2"""
        return 3*x**2 - 2

    def h_double_prime_exact(x):
        """h''(x) = 6x"""
        return 6*x

    h = 0.25
    x_discrete = np.arange(-2, 2 + h, h)
    x_continuous = np.linspace(-2, 2, 1000)

    # First derivatives
    fwd_first = np.array([forward_difference(h_func, x, h) for x in x_discrete[:-1]])
    bwd_first = np.array([backward_difference(h_func, x, h) for x in x_discrete[1:]])
    ctr_first = np.array([centered_difference(h_func, x, h) for x in x_discrete[1:-1]])

    # Second derivatives
    fwd_second = np.array([(h_func(x + 2*h) - 2*h_func(x + h) + h_func(x)) / h**2
                           for x in x_discrete[:-2]])
    bwd_second = np.array([(h_func(x) - 2*h_func(x - h) + h_func(x - 2*h)) / h**2
                           for x in x_discrete[2:]])
    ctr_second = np.array([second_derivative_centered(h_func, x, h)
                           for x in x_discrete[1:-1]])

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First derivative plot
    ax1.plot(x_continuous, h_prime_exact(x_continuous), 'k-', linewidth=2, label='Exact f\'(x)')
    ax1.plot(x_discrete[:-1], fwd_first, 'ro-', markersize=6, label='Forward Difference')
    ax1.plot(x_discrete[1:], bwd_first, 'bs-', markersize=6, label='Backward Difference')
    ax1.plot(x_discrete[1:-1], ctr_first, 'g^-', markersize=6, label='Centered Difference')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f\'(x)', fontsize=12)
    ax1.set_title('First Derivative Approximations', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Second derivative plot
    ax2.plot(x_continuous, h_double_prime_exact(x_continuous), 'k-', linewidth=2,
             label='Exact f\'\'(x)')
    ax2.plot(x_discrete[:-2], fwd_second, 'ro-', markersize=6, label='Forward Difference')
    ax2.plot(x_discrete[2:], bwd_second, 'bs-', markersize=6, label='Backward Difference')
    ax2.plot(x_discrete[1:-1], ctr_second, 'g^-', markersize=6, label='Centered Difference')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f\'\'(x)', fontsize=12)
    ax2.set_title('Second Derivative Approximations', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_4_19_finite_differences.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'problem_4_19_finite_differences.png'")
    print()
    print("Observation: The centered difference approximation is most accurate for both")
    print("first and second derivatives, as it has O(h²) truncation error compared to")
    print("O(h) for forward and backward differences.")
    plt.close()


# =============================================================================
# Problem 4.99: Richardson Extrapolation for f(x)
# =============================================================================

def richardson_first_derivative(func, x, h):
    """
    Richardson extrapolation for first derivative.
    Drich = [4*D(h/2) - D(h)] / 3
    """
    D_h = centered_difference(func, x, h)
    D_h2 = centered_difference(func, x, h/2)
    return (4 * D_h2 - D_h) / 3


def richardson_second_derivative(func, x, h):
    """
    Richardson extrapolation for second derivative.
    Drich = [4*D(h/2) - D(h)] / 3
    """
    D_h = second_derivative_centered(func, x, h)
    D_h2 = second_derivative_centered(func, x, h/2)
    return (4 * D_h2 - D_h) / 3


def problem_4_99():
    """
    Problem 4.99: Richardson extrapolation for f(x) = 25x³ - 6x² + 7x - 88
    at x = 2.067.
    """
    print("\n" + "=" * 80)
    print("Problem 4.99: Richardson Extrapolation for f(x)")
    print("=" * 80)
    print(f"Function: f(x) = 25x³ - 6x² + 7x - 88")
    print(f"Point: x = 2.067")
    print()

    x = 2.067
    h = 0.4

    # First derivative
    print("First Derivative Estimation:")
    print("-" * 60)

    true_first = f_prime(x)
    print(f"Exact f'(2.067) = {true_first:.10f}")
    print()

    D_h = centered_difference(f, x, h)
    D_h2 = centered_difference(f, x, h/2)
    D_rich = richardson_first_derivative(f, x, h)

    print(f"D(h) with h = {h}:")
    print(f"  Value: {D_h:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_h):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_h):.10f}%")
    print()

    print(f"D(h/2) with h = {h/2}:")
    print(f"  Value: {D_h2:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_h2):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_h2):.10f}%")
    print()

    print(f"Richardson Extrapolation D_rich:")
    print(f"  Value: {D_rich:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_rich):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_rich):.10e}%")
    print()

    # Second derivative
    print("Second Derivative Estimation:")
    print("-" * 60)

    true_second = f_double_prime(x)
    print(f"Exact f''(2.067) = {true_second:.10f}")
    print()

    D2_h = second_derivative_centered(f, x, h)
    D2_h2 = second_derivative_centered(f, x, h/2)
    D2_rich = richardson_second_derivative(f, x, h)

    print(f"D(h) with h = {h}:")
    print(f"  Value: {D2_h:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_h):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_h):.10f}%")
    print()

    print(f"D(h/2) with h = {h/2}:")
    print(f"  Value: {D2_h2:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_h2):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_h2):.10f}%")
    print()

    print(f"Richardson Extrapolation D_rich:")
    print(f"  Value: {D2_rich:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_rich):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_rich):.10e}%")
    print()

    print("Discussion:")
    print("Richardson extrapolation significantly improves accuracy by canceling the")
    print("leading error term. The method improves from O(h²) to O(h⁴), reducing error")
    print("by several orders of magnitude compared to standard centered differences.")


# =============================================================================
# Problem 4.99 (Another Equation): Richardson Extrapolation for g(x)
# =============================================================================

def g(x):
    """Function: g(x) = e^x * sin(x)"""
    if isinstance(x, np.ndarray):
        return np.exp(x) * np.sin(x)
    else:
        return exp(x) * sin(x)


def g_prime_exact(x):
    """Exact first derivative: g'(x) = e^x(sin(x) + cos(x))"""
    if isinstance(x, np.ndarray):
        return np.exp(x) * (np.sin(x) + np.cos(x))
    else:
        return exp(x) * (sin(x) + cos(x))


def g_double_prime_exact(x):
    """Exact second derivative: g''(x) = 2e^x cos(x)"""
    if isinstance(x, np.ndarray):
        return 2 * np.exp(x) * np.cos(x)
    else:
        return 2 * exp(x) * cos(x)


def problem_4_99_another():
    """
    Problem 4.99 (Another Equation): Richardson extrapolation for g(x) = e^x sin(x)
    on interval [-2, 2] with h = 0.25, and at x = 2.067.
    """
    print("\n" + "=" * 80)
    print("Problem 4.99 (Another Equation): Richardson Extrapolation for g(x)")
    print("=" * 80)
    print(f"Function: g(x) = e^x sin(x)")
    print()

    # Part 1: Graphical comparison on [-2, 2] with h = 0.25
    print("Part 1: Finite Difference Approximations on [-2, 2]")
    print("-" * 60)

    h = 0.25
    x_discrete = np.arange(-2, 2 + h, h)
    x_continuous = np.linspace(-2, 2, 1000)

    # First derivatives
    fwd_first = np.array([forward_difference(g, x, h) for x in x_discrete[:-1]])
    bwd_first = np.array([backward_difference(g, x, h) for x in x_discrete[1:]])
    ctr_first = np.array([centered_difference(g, x, h) for x in x_discrete[1:-1]])

    # Second derivatives
    fwd_second = np.array([(g(x + 2*h) - 2*g(x + h) + g(x)) / h**2
                           for x in x_discrete[:-2]])
    bwd_second = np.array([(g(x) - 2*g(x - h) + g(x - 2*h)) / h**2
                           for x in x_discrete[2:]])
    ctr_second = np.array([second_derivative_centered(g, x, h)
                           for x in x_discrete[1:-1]])

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First derivative plot
    ax1.plot(x_continuous, g_prime_exact(x_continuous), 'k-', linewidth=2,
             label='Exact g\'(x)')
    ax1.plot(x_discrete[:-1], fwd_first, 'ro-', markersize=5, label='Forward Difference')
    ax1.plot(x_discrete[1:], bwd_first, 'bs-', markersize=5, label='Backward Difference')
    ax1.plot(x_discrete[1:-1], ctr_first, 'g^-', markersize=5, label='Centered Difference')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('g\'(x)', fontsize=12)
    ax1.set_title('First Derivative Approximations for g(x) = e^x sin(x)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Second derivative plot
    ax2.plot(x_continuous, g_double_prime_exact(x_continuous), 'k-', linewidth=2,
             label='Exact g\'\'(x)')
    ax2.plot(x_discrete[:-2], fwd_second, 'ro-', markersize=5, label='Forward Difference')
    ax2.plot(x_discrete[2:], bwd_second, 'bs-', markersize=5, label='Backward Difference')
    ax2.plot(x_discrete[1:-1], ctr_second, 'g^-', markersize=5, label='Centered Difference')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('g\'\'(x)', fontsize=12)
    ax2.set_title('Second Derivative Approximations for g(x) = e^x sin(x)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_4_99_another_finite_differences.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'problem_4_99_another_finite_differences.png'")
    print()

    # Part 2: Richardson extrapolation at x = 2.067
    print("Part 2: Richardson Extrapolation at x = 2.067")
    print("-" * 60)

    x = 2.067
    h = 0.4

    # First derivative
    print("First Derivative Estimation:")
    print()

    true_first = g_prime_exact(x)
    print(f"Exact g'(2.067) = {true_first:.10f}")
    print()

    D_h = centered_difference(g, x, h)
    D_h2 = centered_difference(g, x, h/2)
    D_rich = richardson_first_derivative(g, x, h)

    print(f"D(0.4):")
    print(f"  Value: {D_h:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_h):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_h):.10f}%")
    print()

    print(f"D(0.2):")
    print(f"  Value: {D_h2:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_h2):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_h2):.10f}%")
    print()

    print(f"Richardson Extrapolation D_rich:")
    print(f"  Value: {D_rich:.10f}")
    print(f"  Absolute Error: {abs(true_first - D_rich):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_first, D_rich):.10e}%")
    print()

    # Second derivative
    print("Second Derivative Estimation:")
    print()

    true_second = g_double_prime_exact(x)
    print(f"Exact g''(2.067) = {true_second:.10f}")
    print()

    D2_h = second_derivative_centered(g, x, h)
    D2_h2 = second_derivative_centered(g, x, h/2)
    D2_rich = richardson_second_derivative(g, x, h)

    print(f"D(0.4):")
    print(f"  Value: {D2_h:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_h):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_h):.10f}%")
    print()

    print(f"D(0.2):")
    print(f"  Value: {D2_h2:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_h2):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_h2):.10f}%")
    print()

    print(f"Richardson Extrapolation D_rich:")
    print(f"  Value: {D2_rich:.10f}")
    print(f"  Absolute Error: {abs(true_second - D2_rich):.10e}")
    print(f"  True % Relative Error: {true_percent_relative_error(true_second, D2_rich):.10e}%")
    print()

    print("Discussion:")
    print("For the exponential-trigonometric function g(x) = e^x sin(x), which varies")
    print("more rapidly than polynomials, Richardson extrapolation still provides")
    print("substantial improvement in accuracy. The centered difference is most accurate,")
    print("and Richardson extrapolation further reduces the error significantly.")


# =============================================================================
# ECEN 5080: Log-Log Error Analysis with Richardson Extrapolation
# =============================================================================

def problem_5080():
    """
    ECEN 5080: Compute finite difference approximations for f(x) = sin(x) at x0 = 1
    with step sizes hk = 10^-k for k = 1,...,12. Plot log-log error vs h.
    """
    print("\n" + "=" * 80)
    print("ECEN 5080: Log-Log Error Analysis with Richardson Extrapolation")
    print("=" * 80)
    print(f"Function: f(x) = sin(x)")
    print(f"Point: x0 = 1")
    print(f"Step sizes: hk = 10^-k for k = 1, 2, ..., 12")
    print()

    def func(x):
        if isinstance(x, np.ndarray):
            return np.sin(x)
        else:
            return sin(x)

    x0 = 1
    true_derivative = cos(x0)  # Exact: f'(x) = cos(x), f'(1) = cos(1)

    k_values = np.arange(1, 13)
    h_values = 10.0 ** (-k_values)

    fwd_errors = []
    bwd_errors = []
    ctr_errors = []
    rich_errors = []

    print(f"Exact derivative f'(1) = cos(1) = {true_derivative:.15f}")
    print()

    for k, h in zip(k_values, h_values):
        # Forward difference
        D_fwd = forward_difference(func, x0, h)
        err_fwd = abs(D_fwd - true_derivative)
        fwd_errors.append(err_fwd)

        # Backward difference
        D_bwd = backward_difference(func, x0, h)
        err_bwd = abs(D_bwd - true_derivative)
        bwd_errors.append(err_bwd)

        # Centered difference
        D_ctr = centered_difference(func, x0, h)
        err_ctr = abs(D_ctr - true_derivative)
        ctr_errors.append(err_ctr)

        # Richardson extrapolation
        D_rich = richardson_first_derivative(func, x0, h)
        err_rich = abs(D_rich - true_derivative)
        rich_errors.append(err_rich)

    # Find h that minimizes error for each method
    h_min_fwd = h_values[np.argmin(fwd_errors)]
    h_min_bwd = h_values[np.argmin(bwd_errors)]
    h_min_ctr = h_values[np.argmin(ctr_errors)]
    h_min_rich = h_values[np.argmin(rich_errors)]

    print("Optimal step sizes (minimizing error):")
    print(f"  Forward:    h = {h_min_fwd:.2e}")
    print(f"  Backward:   h = {h_min_bwd:.2e}")
    print(f"  Centered:   h = {h_min_ctr:.2e}")
    print(f"  Richardson: h = {h_min_rich:.2e}")
    print()

    # Create log-log plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.loglog(h_values, fwd_errors, 'ro-', markersize=6, linewidth=2,
              label='Forward Difference')
    ax.loglog(h_values, bwd_errors, 'bs-', markersize=6, linewidth=2,
              label='Backward Difference')
    ax.loglog(h_values, ctr_errors, 'g^-', markersize=6, linewidth=2,
              label='Centered Difference')
    ax.loglog(h_values, rich_errors, 'mv-', markersize=6, linewidth=2,
              label='Richardson Extrapolation')

    # Add reference lines for slopes
    # For coarse h, add slope reference lines
    h_ref = np.array([1e-1, 1e-3])
    slope_1 = 1e-2 * (h_ref / 1e-1)**1  # Slope = 1
    slope_2 = 1e-3 * (h_ref / 1e-1)**2  # Slope = 2
    slope_4 = 1e-5 * (h_ref / 1e-1)**4  # Slope = 4

    ax.loglog(h_ref, slope_1, 'k--', linewidth=1, alpha=0.5, label='Slope = 1 (O(h))')
    ax.loglog(h_ref, slope_2, 'k-.', linewidth=1, alpha=0.5, label='Slope = 2 (O(h²))')
    ax.loglog(h_ref, slope_4, 'k:', linewidth=1, alpha=0.5, label='Slope = 4 (O(h⁴))')

    ax.set_xlabel('Step Size h', fontsize=14)
    ax.set_ylabel('Absolute Error |D_h - f\'(1)|', fontsize=14)
    ax.set_title('Log-Log Error Plot for Finite Difference Approximations', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_5080_loglog_error.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'problem_5080_loglog_error.png'")
    print()

    # Compute empirical slopes for coarse h (using first few points)
    def compute_slope(h_vals, errors, num_points=4):
        """Compute slope in log-log plot using linear regression"""
        log_h = np.log10(h_vals[:num_points])
        log_err = np.log10(errors[:num_points])
        slope = np.polyfit(log_h, log_err, 1)[0]
        return slope

    slope_fwd = compute_slope(h_values, fwd_errors)
    slope_bwd = compute_slope(h_values, bwd_errors)
    slope_ctr = compute_slope(h_values, ctr_errors)
    slope_rich = compute_slope(h_values, rich_errors)

    print("Empirical slopes (for coarse h):")
    print(f"  Forward:    {slope_fwd:.3f} (expected ≈ 1)")
    print(f"  Backward:   {slope_bwd:.3f} (expected ≈ 1)")
    print(f"  Centered:   {slope_ctr:.3f} (expected ≈ 2)")
    print(f"  Richardson: {slope_rich:.3f} (expected ≈ 4)")
    print()

    print("Discussion:")
    print("1. Truncation vs. Round-off Error:")
    print("   - For large h: truncation error dominates, error decreases as h decreases")
    print("   - For small h: round-off error dominates, error increases as h decreases")
    print("   - Optimal h balances these two sources of error")
    print()
    print("2. Richardson Extrapolation Benefits:")
    print("   - Improves convergence rate from O(h²) to O(h⁴)")
    print("   - Provides much smaller errors for moderate h values")
    print("   - However, for very small h, round-off errors limit improvement")
    print("   - Richardson is most beneficial when h is not too small (before round-off dominates)")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all problems"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  ECEN 4293 Project 1: Taylor Series and Finite-Difference Approximations  ".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    # Run all problems
    problem_4_12()
    problem_4_15()
    problem_4_16()
    problem_4_18()
    problem_4_19()
    problem_4_99()
    problem_4_99_another()
    problem_5080()

    print("\n" + "=" * 80)
    print("All problems completed successfully!")
    print("Generated plots:")
    print("  - problem_4_12_taylor_series.png")
    print("  - problem_4_19_finite_differences.png")
    print("  - problem_4_99_another_finite_differences.png")
    print("  - problem_5080_loglog_error.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
