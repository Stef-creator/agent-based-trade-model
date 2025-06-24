import numpy as np
import matplotlib.pyplot as plt

def kappa_sensitvity():
    kappa_values = np.linspace(0, 1, 11)
    CVs = []
    sample_Y = {}

# --- Parameters ---
    T = 200
    num_economies = 5
    firms_per_economy = 20
    mu = 0.6
    iota = 0.4
    rho = 0.6
    alpha = 0.375
    beta = 0.5
    gamma = 1.0
    phi = 1.0
    lambda_exo = 0.05
    theta_innovator = 0.0
    theta_imitator = 0.0
    sigma = 0.1
    chi = 0.5
    psi = 0.0   # No spillover
    ex_rate_shock_var = 0.0001

    for kappa in kappa_values:
        np.random.seed(42)  # Consistent random seed for each run

        a_firm = np.ones((num_economies, firms_per_economy))
        A_firm = np.ones((num_economies, firms_per_economy))
        z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
        K_firm = np.ones((num_economies, firms_per_economy))
        Y_firm = np.ones((num_economies, firms_per_economy))
        Pi_firm = np.ones((num_economies, firms_per_economy))
        I_firm = np.zeros((num_economies, firms_per_economy))
        R_firm = np.zeros((num_economies, firms_per_economy))
        innovators = (np.arange(firms_per_economy) % 2 == 0)
        Y_economy = np.full(num_economies, 100.0)
        A_economy = np.ones(num_economies)
        E_economy = np.ones(num_economies)
        w_economy = np.full(num_economies, 5.0)
        Yw_prev1 = np.full(num_economies, 401.0)
        Yw_prev2 = np.full(num_economies, 401.0)
        Y_exo = 1.0
        ex_rate = np.ones(num_economies)
        exports = np.ones(num_economies)
        exports_prev = np.ones(num_economies)
        exports_global = exports.sum()
        exports_global_prev = exports_global

        Y_hist = [[] for _ in range(num_economies)]

        for t in range(T):
            z_economy = z_firm.sum(axis=1)
            A_prev = A_economy.copy()
            A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

            exports_prev[:] = exports[:]
            for j in range(num_economies):
                Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
                exports[j] = (Yw_t ** alpha) * z_economy[j]

            exports_global_prev = exports_global
            exports_global = exports.sum()
            global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

            for j in range(num_economies):
                own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
                rel_growth = own_growth - global_growth
                shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
                ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock

            price_firm = (1 + mu) * w_economy[:, None] / A_firm
            E_firm = 1 / (price_firm * ex_rate[:, None])
            E_economy = (z_firm * E_firm).sum(axis=1) / z_economy

            growth_A = (A_economy / A_prev - 1)
            w_economy *= (1 + gamma * growth_A)

            rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
            e_term = 1.0 / (1 - z_economy)
            E_global = (E_economy * z_economy).sum()
            phi_term = phi / beta * e_term * (E_economy / E_global - 1)
            alpha_term = alpha / beta * rel_Yw
            Y_economy *= (1 + alpha_term + phi_term)

            for j in range(num_economies):
                for i in range(firms_per_economy):
                    Y_firm[j, i] = z_firm[j, i] / z_economy[j] * Y_economy[j]
                    Pi_firm[j, i] = mu * w_economy[j] / A_firm[j, i] * Y_firm[j, i]
                    I_firm[j, i] = min(iota * Y_firm[j, i], Pi_firm[j, i])
                    R_firm[j, i] = max(0.0, min(rho * Y_firm[j, i], Pi_firm[j, i] - I_firm[j, i]))
                    prob_success = R_firm[j, i] / Y_firm[j, i] + (theta_innovator if innovators[i] else theta_imitator)
                    if np.random.rand() < min(1.0, prob_success):
                        if innovators[i]:
                            delta = np.random.normal(0, sigma)
                        else:
                            gap = max(chi * (a_firm.mean() - a_firm[j, i]), 0)
                            delta = np.random.normal(0, gap) if gap > 0 else 0
                        a_firm[j, i] = max(a_firm[j, i] + delta, a_firm[j, i])

            for j in range(num_economies):
                for i in range(firms_per_economy):
                    denom = K_firm[j, i] + I_firm[j, i]
                    if denom > 0:
                        A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                    K_firm[j, i] += I_firm[j, i]

            for j in range(num_economies):
                for i in range(firms_per_economy):
                    rel_fit = E_firm[j, i] / E_global
                    z_firm[j, i] *= (1 + phi * (rel_fit - 1))

            Yw_prev2 = Yw_prev1.copy()
            Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
            Y_exo *= (1 + lambda_exo)

            for j in range(num_economies):
                Y_hist[j].append(Y_economy[j])

        final_Y = np.array([Y_hist[j][-1] for j in range(num_economies)])
        CV = np.std(final_Y) / np.mean(final_Y)
        CVs.append(CV)

        if np.isclose(kappa, 0.0) or np.isclose(kappa, 0.02) or np.isclose(kappa, 0.05):
            sample_Y[round(kappa, 3)] = Y_hist.copy()

# --- Plot κ vs. CV ---
    plt.figure(figsize=(7,5))
    plt.plot(kappa_values, CVs, marker='o')
    plt.xlabel('κ (FX Sensitivity)')
    plt.ylabel('Coefficient of Variation of GDP (final step)')
    plt.title('Divergence vs. Exchange Rate Feedback (No Spillover)')
    plt.grid(True)
    plt.show()

# --- Plot sample GDPs for selected kappas ---
    for kappa, Y_hist in sample_Y.items():
        plt.figure(figsize=(8,5))
        for j in range(num_economies):
            plt.plot(Y_hist[j], label=f"Economy {j+1}")
        plt.title(f"GDP Trajectories, κ = {kappa}")
        plt.xlabel('Time')

