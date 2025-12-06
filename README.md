# MonteCarloOptionEngine

A modern **C++17 Monte Carlo pricing engine** for European and exotic options under the Black–Scholes model, featuring:

- ✔ **Antithetic variance reduction**
- ✔ **Pathwise estimators for Delta & Vega**
- ✔ **Asian up-and-out barrier option pricer**
- ✔ **Sobol-style (van der Corput) low-discrepancy quasi-random normals**
- ✔ **Comparisons against closed-form Black–Scholes**
- ✔ **95% confidence intervals and runtime benchmarking**
- ✔ **OpenMP-ready loop structure for multi-threaded acceleration**

This project was implemented as part of my preparation for quantitative finance internships.

---

## Features

### **1. European Calls and Puts (Monte Carlo)**
- Geometric Brownian Motion exact simulation  
- Antithetic variates  
- Confidence intervals  
- Black–Scholes analytic comparison  

### **2. Greeks (Pathwise Estimators)**
The call option pricer computes:
- **Delta**
- **Vega**
- And corresponding **standard errors**

### **3. Asian Up-and-Out Barrier Option**
- Arithmetic averaging  
- Knock-out barrier monitored at discrete time steps  
- Monte Carlo with arbitrary number of paths and steps  

### **4. Low-Discrepancy “Sobol-like” Quasi-Random Numbers**
Implements 1D Sobol via **van der Corput base-2 radical inversion**,  
transformed into a normal distribution using the Box–Muller method.

### **5. OpenMP-ready Parallelism**
All Monte Carlo loops can be parallelised by uncommenting:

```cpp
#pragma omp parallel for reduction(+:...)
