# python code for coverage prob 11
import math
import numpy as np

# Function to compute the 95% exact lower bound for OR (solve Pθ(A >= a) = 0.025)
def solve_lower_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl, alpha=0.05):
    target = alpha / 2  # 0.025 for each tail
    t1 = a + c  # total exposed individuals
    # Define a function for upper-tail probability Pθ(A >= a) under odds ratio = θ
    def P_ge(theta):
        # Compute normalized probability that A >= a
        norm = 0.0
        tail = 0.0
        # Possible range of A given margins:
        i_min = max(0, t1 - n_ctrl)
        i_max = min(t1, n_case)
        for j in range(i_min, i_max + 1):
            term = comb_n_case[j] * comb_n_ctrl[t1 - j] * (theta ** j)
            norm += term
            if j >= a:
                tail += term
        return tail / norm if norm > 0 else 0.0

    p_at1 = P_ge(1.0)
    # Determine search direction: compare P(A >= a) at θ=1 to 0.025
    if p_at1 > target:
        # At θ=1 the upper-tail is too high; need smaller θ (θ < 1) to lower P(A>=a)
        lo, hi = 0.0, 1.0
    else:
        # At θ=1 the upper-tail is already low; need larger θ (>1) to reach 0.025
        lo, hi = 1.0, 1.0
        # Increase hi until P_ge(hi) >= 0.025 (or cap at a large number)
        while P_ge(hi) < target and hi < 1e6:
            hi *= 2
        if P_ge(hi) < target:
            return float('inf')  # Odds ratio needs to be effectively infinite
    # Binary search between lo and hi for θ that gives P_ge ≈ 0.025
    for _ in range(30):  # 30 iterations for high precision
        mid = (lo + hi) / 2
        p_mid = P_ge(mid)
        if p_mid >= target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# Function to compute the 95% exact upper bound for OR (solve Pθ(A <= a) = 0.025)
def solve_upper_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl, alpha=0.05):
    target = alpha / 2  # 0.025
    t1 = a + c  # total exposed
    def P_le(theta):
        # Compute normalized probability that A <= a
        norm = 0.0
        cum = 0.0
        i_min = max(0, t1 - n_ctrl)
        i_max = min(t1, n_case)
        for j in range(i_min, i_max + 1):
            term = comb_n_case[j] * comb_n_ctrl[t1 - j] * (theta ** j)
            norm += term
            if j <= a:
                cum += term
        return cum / norm if norm > 0 else 0.0

    p_at1 = P_le(1.0)
    # Determine search direction based on P(A <= a) at θ=1
    if p_at1 > target:
        # Lower-tail is too high at θ=1; need larger θ to reduce P(A<=a)
        lo, hi = 1.0, 1.0
        while P_le(hi) > target and hi < 1e6:
            hi *= 2
        if P_le(hi) > target:
            return float('inf')  # OR -> infinite to push lower-tail down
    else:
        # Lower-tail is already <= 0.025 at θ=1; need smaller θ to raise P(A<=a)
        lo, hi = 0.0, 1.0
    # Binary search between lo and hi for θ giving P_le ≈ 0.025
    for _ in range(30):
        mid = (lo + hi) / 2
        p_mid = P_le(mid)
        if p_mid <= target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# Define simulation scenarios: (n_case, n_ctrl, exposure_probability)
scenarios = [
    (20, 20, 0.1),
    (20, 20, 0.5),
    (20, 40, 0.1),
    (20, 40, 0.5),
    (30, 30, 0.1),
    (30, 30, 0.5),
    (40, 20, 0.1),
    (40, 20, 0.5),
    (40, 40, 0.1),
    (40, 40, 0.5),
]

np.random.seed(0)  # for reproducibility
num_sim = 5000
print("n_case  n_ctrl   p_exposure   Coverage")
for n_case, n_ctrl, p in scenarios:
    # Pre-compute combinatorial coefficients for efficiency
    comb_n_case = [math.comb(n_case, k) for k in range(n_case + 1)]
    comb_n_ctrl = [math.comb(n_ctrl, k) for k in range(n_ctrl + 1)]
    cover_count = 0
    for _ in range(num_sim):
        # Simulate one 2x2 table under OR=1
        a = np.random.binomial(n_case, p)    # exposed cases
        c = np.random.binomial(n_ctrl, p)    # exposed controls
        b = n_case - a                       # unexposed cases
        d = n_ctrl - c                       # unexposed controls

        # Compute 95% exact CI for OR:
        if (a == 0 and c == 0) or (b == 0 and d == 0):
            # Degenerate case: no variation in exposure (CI is [0, ∞], always covers 1)
            L, U = 0.0, float('inf')
        elif a == 0 or d == 0:
            # Odds ratio = 0 (a=0 or d=0 → numerator zero)
            L = 0.0
            U = solve_upper_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl)
        elif b == 0 or c == 0:
            # Odds ratio = ∞ (b=0 or c=0 → denominator zero)
            L = solve_lower_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl)
            U = float('inf')
        else:
            # General case: finite OR estimate
            L = solve_lower_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl)
            U = solve_upper_bound(a, c, n_case, n_ctrl, comb_n_case, comb_n_ctrl)
        # Check if true OR=1 is within the interval [L, U]
        if L <= 1.0 <= U:
            cover_count += 1
    coverage = cover_count / num_sim
    print(f"{n_case:5d}   {n_ctrl:5d}       {p:<9.2f}  {coverage:.3f}")
