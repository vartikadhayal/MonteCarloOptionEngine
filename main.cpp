#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

// -----------------------------
// basic data structures
// -----------------------------

struct OptionParams {
    double S0;    // initial spot
    double K;     // strike
    double r;     // risk-free rate
    double sigma; // volatility
    double T;     // maturity in years
};

enum class OptionType { Call, Put };

struct MCResult {
    double price{};
    double stderr{};
    double ci_low{};
    double ci_high{};
    double time_ms{};

    // greeks (used for call with pathwise estimators)
    double delta{};
    double delta_stderr{};
    double vega{};
    double vega_stderr{};
};

// -----------------------------
// constants / utilities
// -----------------------------

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double black_scholes_price(const OptionParams &p, OptionType type) {
    double S0 = p.S0;
    double K = p.K;
    double r = p.r;
    double sigma = p.sigma;
    double T = p.T;

    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) /
                (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    if (type == OptionType::Call) {
        return S0 * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    } else {
        return K * std::exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1);
    }
}

// -----------------------------
// low-discrepancy (sobol-style) sequence
// -----------------------------
//
// 1D sobol reduces to van der corput in base 2
// using that + box–muller to get quasi-random normals
//

double van_der_corput_base2(unsigned int n) {
    double v = 0.0;
    double denom = 1.0;
    while (n > 0U) {
        denom *= 2.0;
        unsigned int bit = n & 1U;
        v += static_cast<double>(bit) / denom;
        n >>= 1;
    }
    return v; // in [0,1)
}

double sobol_like_normal_from_index(unsigned int i) {
    double u1 = van_der_corput_base2(i + 1U);
    double u2 = van_der_corput_base2(i + 123457U);

    if (u1 <= 0.0) u1 = 1e-12;
    if (u2 <= 0.0) u2 = 1e-12;

    double R = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;

    double z = R * std::cos(theta); // standard normal
    return z;
}

// -----------------------------
// monte carlo european pricer
// -----------------------------

class MonteCarloPricer {
public:
    MonteCarloPricer(const OptionParams &params, std::uint64_t seed = 42ULL)
        : params_(params), base_seed_(seed) {}

    // call with:
    // - antithetic variates
    // - pathwise delta & vega
    MCResult price_call_with_greeks(std::size_t paths,
                                    bool use_antithetic = true) {
        if (paths < 2) {
            throw std::runtime_error("Need at least 2 paths.");
        }

        const double S0 = params_.S0;
        const double K = params_.K;
        const double r = params_.r;
        const double sigma = params_.sigma;
        const double T = params_.T;

        const double sqrtT = std::sqrt(T);
        const double discount = std::exp(-r * T);

        auto start = std::chrono::high_resolution_clock::now();

        std::mt19937_64 rng(base_seed_);
        std::normal_distribution<double> normal(0.0, 1.0);

        double sum_payoff = 0.0;
        double sum_payoff_sq = 0.0;

        double sum_delta = 0.0;
        double sum_delta_sq = 0.0;

        double sum_vega = 0.0;
        double sum_vega_sq = 0.0;

        std::size_t effective_paths =
            use_antithetic ? (paths / 2) : paths;
        if (effective_paths == 0) {
            effective_paths = 1;
        }

        // trivially parallelisable loop:
        // #pragma omp parallel for reduction(+:sum_payoff,sum_payoff_sq,sum_delta,sum_delta_sq,sum_vega,sum_vega_sq)
        // for (std::size_t i = 0; i < effective_paths; ++i) { ... }
        // kept it single-threaded

        for (std::size_t i = 0; i < effective_paths; ++i) {
            double Z = normal(rng);

            // first path (Z)
            double ST1 = S0 * std::exp((r - 0.5 * sigma * sigma) * T +
                                       sigma * sqrtT * Z);
            double payoff1 = std::max(ST1 - K, 0.0);
            double indicator1 = (ST1 > K) ? 1.0 : 0.0;

            double delta_contrib1 = discount * (ST1 / S0) * indicator1;
            double vega_contrib1 =
                discount * ST1 * (-sigma * T + sqrtT * Z) * indicator1;

            if (use_antithetic) {
                double Z2 = -Z;
                double ST2 = S0 * std::exp((r - 0.5 * sigma * sigma) * T +
                                           sigma * sqrtT * Z2);
                double payoff2 = std::max(ST2 - K, 0.0);
                double indicator2 = (ST2 > K) ? 1.0 : 0.0;

                double delta_contrib2 =
                    discount * (ST2 / S0) * indicator2;
                double vega_contrib2 =
                    discount * ST2 * (-sigma * T + sqrtT * Z2) * indicator2;

                double payoff_pair = 0.5 * (payoff1 + payoff2);
                double delta_pair = 0.5 * (delta_contrib1 + delta_contrib2);
                double vega_pair = 0.5 * (vega_contrib1 + vega_contrib2);

                double discounted_payoff = discount * payoff_pair;

                sum_payoff += discounted_payoff;
                sum_payoff_sq += discounted_payoff * discounted_payoff;

                sum_delta += delta_pair;
                sum_delta_sq += delta_pair * delta_pair;

                sum_vega += vega_pair;
                sum_vega_sq += vega_pair * vega_pair;
            } else {
                double discounted_payoff = discount * payoff1;

                sum_payoff += discounted_payoff;
                sum_payoff_sq += discounted_payoff * discounted_payoff;

                sum_delta += delta_contrib1;
                sum_delta_sq += delta_contrib1 * delta_contrib1;

                sum_vega += vega_contrib1;
                sum_vega_sq += vega_contrib1 * vega_contrib1;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        double N = static_cast<double>(effective_paths);

        double mean_payoff = sum_payoff / N;
        double mean_payoff_sq = sum_payoff_sq / N;
        double var_payoff = mean_payoff_sq - mean_payoff * mean_payoff;
        if (var_payoff < 0.0) var_payoff = 0.0;

        double mean_delta = sum_delta / N;
        double mean_delta_sq = sum_delta_sq / N;
        double var_delta = mean_delta_sq - mean_delta * mean_delta;
        if (var_delta < 0.0) var_delta = 0.0;

        double mean_vega = sum_vega / N;
        double mean_vega_sq = sum_vega_sq / N;
        double var_vega = mean_vega_sq - mean_vega * mean_vega;
        if (var_vega < 0.0) var_vega = 0.0;

        double stderr_price = std::sqrt(var_payoff / N);
        double stderr_delta = std::sqrt(var_delta / N);
        double stderr_vega = std::sqrt(var_vega / N);

        double z95 = 1.96;
        double ci_low = mean_payoff - z95 * stderr_price;
        double ci_high = mean_payoff + z95 * stderr_price;

        MCResult res;
        res.price = mean_payoff;
        res.stderr = stderr_price;
        res.ci_low = ci_low;
        res.ci_high = ci_high;
        res.time_ms = elapsed_ms;
        res.delta = mean_delta;
        res.delta_stderr = stderr_delta;
        res.vega = mean_vega;
        res.vega_stderr = stderr_vega;

        return res;
    }

    // put, using antithetic variates but no greeks
    MCResult price_put(std::size_t paths, bool use_antithetic = true) {
        if (paths < 2) {
            throw std::runtime_error("Need at least 2 paths.");
        }

        const double S0 = params_.S0;
        const double K = params_.K;
        const double r = params_.r;
        const double sigma = params_.sigma;
        const double T = params_.T;

        const double sqrtT = std::sqrt(T);
        const double discount = std::exp(-r * T);

        auto start = std::chrono::high_resolution_clock::now();

        std::mt19937_64 rng(base_seed_ + 1ULL);
        std::normal_distribution<double> normal(0.0, 1.0);

        double sum_payoff = 0.0;
        double sum_payoff_sq = 0.0;

        std::size_t effective_paths =
            use_antithetic ? (paths / 2) : paths;
        if (effective_paths == 0) {
            effective_paths = 1;
        }

        // also OpenMP-ready:
        // #pragma omp parallel for reduction(+:sum_payoff,sum_payoff_sq)
        for (std::size_t i = 0; i < effective_paths; ++i) {
            double Z = normal(rng);

            double ST1 = S0 * std::exp((r - 0.5 * sigma * sigma) * T +
                                       sigma * sqrtT * Z);
            double payoff1 = std::max(K - ST1, 0.0);

            if (use_antithetic) {
                double Z2 = -Z;
                double ST2 = S0 * std::exp((r - 0.5 * sigma * sigma) * T +
                                           sigma * sqrtT * Z2);
                double payoff2 = std::max(K - ST2, 0.0);

                double payoff_pair = 0.5 * (payoff1 + payoff2);
                double discounted = discount * payoff_pair;
                sum_payoff += discounted;
                sum_payoff_sq += discounted * discounted;
            } else {
                double discounted = discount * payoff1;
                sum_payoff += discounted;
                sum_payoff_sq += discounted * discounted;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        double N = static_cast<double>(effective_paths);
        double mean_payoff = sum_payoff / N;
        double mean_payoff_sq = sum_payoff_sq / N;
        double var_payoff = mean_payoff_sq - mean_payoff * mean_payoff;
        if (var_payoff < 0.0) var_payoff = 0.0;

        double stderr_price = std::sqrt(var_payoff / N);

        double z95 = 1.96;
        double ci_low = mean_payoff - z95 * stderr_price;
        double ci_high = mean_payoff + z95 * stderr_price;

        MCResult res;
        res.price = mean_payoff;
        res.stderr = stderr_price;
        res.ci_low = ci_low;
        res.ci_high = ci_high;
        res.time_ms = elapsed_ms;

        return res;
    }

private:
    OptionParams params_;
    std::uint64_t base_seed_;
};

// -----------------------------
// asian up-and-out barrier call
// -----------------------------

MCResult price_asian_up_and_out_call(
    const OptionParams &p,
    std::size_t paths,
    std::size_t steps,
    double barrier
) {
    if (paths < 2 || steps < 1) {
        throw std::runtime_error("Need at least 2 paths and 1 step.");
    }

    const double S0 = p.S0;
    const double K = p.K;
    const double r = p.r;
    const double sigma = p.sigma;
    const double T = p.T;

    const double dt = T / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);
    const double discount = std::exp(-r * T);

    auto start = std::chrono::high_resolution_clock::now();

    std::mt19937_64 rng(123456789ULL);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (std::size_t i = 0; i < paths; ++i) {
        double St = S0;
        double sum_S = 0.0;
        bool knocked_out = false;

        for (std::size_t n = 0; n < steps; ++n) {
            double Z = normal(rng);
            St = St * std::exp((r - 0.5 * sigma * sigma) * dt
                               + sigma * sqrt_dt * Z);

            if (St >= barrier) {
                knocked_out = true;
                break;
            }

            sum_S += St;
        }

        double payoff = 0.0;
        if (!knocked_out) {
            double avg_S = sum_S / static_cast<double>(steps);
            payoff = std::max(avg_S - K, 0.0);
        }

        double discounted = discount * payoff;
        sum_payoff += discounted;
        sum_payoff_sq += discounted * discounted;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    double N = static_cast<double>(paths);
    double mean_payoff = sum_payoff / N;
    double mean_payoff_sq = sum_payoff_sq / N;
    double var_payoff = mean_payoff_sq - mean_payoff * mean_payoff;
    if (var_payoff < 0.0) var_payoff = 0.0;

    double stderr_price = std::sqrt(var_payoff / N);

    double z95 = 1.96;
    double ci_low = mean_payoff - z95 * stderr_price;
    double ci_high = mean_payoff + z95 * stderr_price;

    MCResult res;
    res.price = mean_payoff;
    res.stderr = stderr_price;
    res.ci_low = ci_low;
    res.ci_high = ci_high;
    res.time_ms = elapsed_ms;

    return res;
}

// -----------------------------
// main
// -----------------------------

int main() {
    std::cout << "Monte Carlo European Option Pricer (Black-Scholes)\n";
    std::cout << "Expert version: antithetic variates + pathwise Greeks\n";
    std::cout << "Extended: Asian barrier + Sobol-style quasi-randoms\n";
    std::cout << "-----------------------------------------------------\n\n";

    OptionParams params;
    params.S0 = 100.0;
    params.K = 100.0;
    params.r = 0.02;
    params.sigma = 0.20;
    params.T = 1.0;

    std::size_t paths;
    std::cout << "Enter number of Monte Carlo paths for European option (e.g. 200000): ";
    if (!(std::cin >> paths) || paths < 1000) {
        std::cout << "Using default: 200000 paths.\n";
        paths = 200000;
    }

    bool use_antithetic = true;

    MonteCarloPricer pricer(params);

    MCResult call_res = pricer.price_call_with_greeks(paths, use_antithetic);
    MCResult put_res = pricer.price_put(paths, use_antithetic);

    double bs_call = black_scholes_price(params, OptionType::Call);
    double bs_put = black_scholes_price(params, OptionType::Put);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nParameters (European options):\n";
    std::cout << "  S0    = " << params.S0 << "\n";
    std::cout << "  K     = " << params.K << "\n";
    std::cout << "  r     = " << params.r << "\n";
    std::cout << "  sigma = " << params.sigma << "\n";
    std::cout << "  T     = " << params.T << " years\n";
    std::cout << "  Paths = " << paths << "\n";
    std::cout << "  Antithetic variates: " << (use_antithetic ? "ON" : "OFF") << "\n\n";

    std::cout << "Call option (Monte Carlo):\n";
    std::cout << "  Price       = " << call_res.price << "\n";
    std::cout << "  Std error   = " << call_res.stderr << "\n";
    std::cout << "  95% CI      = [" << call_res.ci_low << ", "
              << call_res.ci_high << "]\n";
    std::cout << "  Runtime     = " << call_res.time_ms << " ms\n\n";

    std::cout << "Call Greeks (pathwise estimators):\n";
    std::cout << "  Delta       = " << call_res.delta
              << "  (SE = " << call_res.delta_stderr << ")\n";
    std::cout << "  Vega        = " << call_res.vega
              << "  (SE = " << call_res.vega_stderr << ")\n\n";

    std::cout << "Put option (Monte Carlo):\n";
    std::cout << "  Price       = " << put_res.price << "\n";
    std::cout << "  Std error   = " << put_res.stderr << "\n";
    std::cout << "  95% CI      = [" << put_res.ci_low << ", "
              << put_res.ci_high << "]\n";
    std::cout << "  Runtime     = " << put_res.time_ms << " ms\n\n";

    std::cout << "Closed-form Black-Scholes prices:\n";
    std::cout << "  Call (BS)   = " << bs_call << "\n";
    std::cout << "  Put  (BS)   = " << bs_put << "\n\n";

    std::cout << "Difference MC - BS (call): " << (call_res.price - bs_call) << "\n";
    std::cout << "Difference MC - BS (put) : " << (put_res.price - bs_put) << "\n\n";

    // asian up-and-out barrier
    std::size_t asian_paths;
    std::size_t asian_steps;
    double barrier;

    std::cout << "Pricing an Asian up-and-out barrier call...\n";
    std::cout << "Enter number of paths for Asian option (e.g. 100000): ";
    if (!(std::cin >> asian_paths) || asian_paths < 1000) {
        std::cout << "Using default: 100000 paths.\n";
        asian_paths = 100000;
    }

    std::cout << "Enter number of time steps per path (e.g. 252): ";
    if (!(std::cin >> asian_steps) || asian_steps < 1) {
        std::cout << "Using default: 252 steps.\n";
        asian_steps = 252;
    }

    std::cout << "Enter barrier level (e.g. 130): ";
    if (!(std::cin >> barrier) || barrier <= params.S0) {
        std::cout << "Using default barrier: 130.0\n";
        barrier = 130.0;
    }

    MCResult asian_res = price_asian_up_and_out_call(
        params, asian_paths, asian_steps, barrier
    );

    std::cout << "\nAsian up-and-out barrier call:\n";
    std::cout << "  Barrier     = " << barrier << "\n";
    std::cout << "  Paths       = " << asian_paths << "\n";
    std::cout << "  Steps/path  = " << asian_steps << "\n";
    std::cout << "  Price       = " << asian_res.price << "\n";
    std::cout << "  Std error   = " << asian_res.stderr << "\n";
    std::cout << "  95% CI      = [" << asian_res.ci_low << ", "
              << asian_res.ci_high << "]\n";
    std::cout << "  Runtime     = " << asian_res.time_ms << " ms\n\n";

    // sobol-style demo
    std::cout << "First 5 Sobol-style quasi-random normals (van der Corput base-2):\n";
    for (unsigned int i = 0; i < 5U; ++i) {
        double z = sobol_like_normal_from_index(i);
        std::cout << "  z[" << i << "] = " << z << "\n";
    }

    std::cout << "\nDone.\n";

    return 0;
}
