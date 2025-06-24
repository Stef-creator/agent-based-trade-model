# === Llerena & Lorentz (2004) ABM - Procedural Baseline ===
# Replicates the model structure in full detail, then all models after are incremental extension

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def run_ll_abm(
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,           # Investment rate (share of output)
    rho=0.6,            # Share of output potentially for R&D
    alpha=0.375,        # Income elasticity of exports
    beta=0.5,           # Income elasticity of imports
    gamma=1.0,          # Wage adjustment to productivity growth (gamma=1: fully absorbed)
    phi=1.0,            # Price elasticity of market share selection
    lambda_exo=0.05,    # Exogenous world income growth rate
    theta_innovator=0.2,
    theta_imitator=0.2,
    sigma=0.1,          # Innovator's technological opportunity (std)
    chi=0.5,            # Imitator's absorptive capacity
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    seed=None
):  
    """
    Runs a multi-economy agent-based model (ABM) simulating firm-level learning, innovation, and macroeconomic dynamics.

    Parameters
    ----------
    T : int, optional
        Number of time steps (periods) to simulate (default: 200).
    num_economies : int, optional
        Number of economies in the simulation (default: 5).
    firms_per_economy : int, optional
        Number of firms in each economy (default: 20).
    mu : float, optional
        Markup rate for firm pricing (default: 0.6).
    iota : float, optional
        Investment rate (share of output invested by firms, default: 0.4).
    rho : float, optional
        Maximum share of output allocated to R&D (default: 0.6).
    alpha : float, optional
        Income elasticity of exports (default: 0.375).
    beta : float, optional
        Income elasticity of imports (default: 0.5).
    gamma : float, optional
        Wage adjustment parameter to productivity growth (default: 1.0).
    phi : float, optional
        Price elasticity of market share selection (default: 1.0).
    lambda_exo : float, optional
        Exogenous world income growth rate (default: 0.05).
    theta_innovator : float, optional
        Exogenous boost to R&D success probability for innovators (default: 0.2).
    theta_imitator : float, optional
        Exogenous boost to R&D success probability for imitators (default: 0.2).
    sigma : float, optional
        Standard deviation of technological opportunity for innovators (default: 0.1).
    chi : float, optional
        Imitator's absorptive capacity (default: 0.5).
    Y0 : float, optional
        Initial GDP per economy (default: 100.0).
    w0 : float, optional
        Initial wage per economy (default: 5.0).
    Yw0 : float, optional
        Initial rest-of-world GDP (default: 401.0).
    Y_exo0 : float, optional
        Initial exogenous world demand component (default: 1.0).
    seed : int or None, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    dict
        Dictionary containing simulation results and histories:
            - "Y_hist": List of GDP histories for each economy.
            - "A_hist": List of average effective productivity histories for each economy.
            - "E_global_hist": List of global competitiveness values over time.
            - "a_global_hist": List of global productivity values over time.
            - "final_Y": Final GDP values for each economy.
            - "final_A": Final average effective productivity for each economy.
            - "final_w": Final wage values for each economy.
            - "final_a_firm": Final productivity values for each firm.
            - "final_A_firm": Final capital-weighted productivity for each firm.
            - "final_z_firm": Final market share for each firm.
            - "final_K_firm": Final capital stock for each firm.

    Notes
    -----
    The function also generates plots of key macroeconomic and productivity variables over time.
    The model implements firm-level innovation and imitation, market share dynamics, and macroeconomic feedbacks.
    """
    if seed is not None:
        np.random.seed(seed)

    # === Initialization ===
    a_firm = np.ones((num_economies, firms_per_economy))                       
    A_firm = np.ones((num_economies, firms_per_economy))                       
    z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
    K_firm = np.ones((num_economies, firms_per_economy))                       

    Y_firm = np.ones((num_economies, firms_per_economy))
    Pi_firm = np.ones((num_economies, firms_per_economy))
    I_firm = np.zeros((num_economies, firms_per_economy))
    R_firm = np.zeros((num_economies, firms_per_economy))

    innovators = (np.arange(firms_per_economy) % 2 == 0)  # Half of all firms are innovators

    Y_economy = np.full(num_economies, Y0)
    A_economy = np.ones(num_economies)
    E_economy = np.ones(num_economies)
    w_economy = np.full(num_economies, w0)

    Yw_prev1 = np.full(num_economies, Yw0)  # Lagged rest-of-world GDP
    Yw_prev2 = np.full(num_economies, Yw0)
    Y_exo = Y_exo0  # Exogenous world demand component

    # === Storage for analysis ===
    Y_hist = [[] for _ in range(num_economies)]
    A_hist = [[] for _ in range(num_economies)]
    E_global_hist = []
    a_global_hist = []

    # === Simulation Loop ===
    for t in range(T):
        # -- Macro variables --
        z_economy = z_firm.sum(axis=1)  # Total market share per economy
        A_prev = A_economy.copy()
        # (Weighted) average effective productivity per economy
        A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

        # Price and Efficiency
        price_firm = (1 + mu) * w_economy[:, None] / A_firm  # Eq (13)
        E_firm = 1 / price_firm                              # Competitiveness
        E_economy = (z_firm * E_firm).sum(axis=1) / z_economy
        # Wages (macro, wage regime)
        growth_A = (A_economy / A_prev - 1)
        w_economy *= (1 + gamma * growth_A)                  # Eq (10)

        # Output Update (GDP, Eq 9)
        rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
        e_term = 1.0 / (1 - z_economy)
        E_global = (E_economy * z_economy).sum()
        phi_term = phi / beta * e_term * (E_economy / E_global - 1)
        alpha_term = alpha / beta * rel_Yw
        Y_economy *= (1 + alpha_term + phi_term)

        # -- Firm-level updates --
        for j in range(num_economies):
            for i in range(firms_per_economy):
                # Output allocation (market share, Eq 11)
                Y_firm[j, i] = z_firm[j, i] / z_economy[j] * Y_economy[j]
                # Profit (markup, Eq 14)
                Pi_firm[j, i] = mu * w_economy[j] / A_firm[j, i] * Y_firm[j, i]
                # Investment (Eq 15)
                I_firm[j, i] = min(iota * Y_firm[j, i], Pi_firm[j, i])
                # R&D (Eq 16)
                R_firm[j, i] = max(0.0, min(rho * Y_firm[j, i], Pi_firm[j, i] - I_firm[j, i]))
                # Probability of R&D success (Eq 16, plus exogenous boost for innovator/imitator)
                prob_success = R_firm[j, i] / Y_firm[j, i] + (theta_innovator if innovators[i] else theta_imitator)
                if np.random.rand() < min(1.0, prob_success):
                    if innovators[i]:
                        delta = np.random.normal(0, sigma)  # Eq 17 (innovator)
                    else:
                        gap = max(chi * (a_firm.mean() - a_firm[j, i]), 0)
                        delta = np.random.normal(0, gap) if gap > 0 else 0  # Eq 18 (imitator)
                    a_firm[j, i] = max(a_firm[j, i] + delta, a_firm[j, i])  # Eq 19
                # Update capital-weighted productivity (Eq 12)
                denom = K_firm[j, i] + I_firm[j, i]
                if denom > 0:
                    A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                K_firm[j, i] += I_firm[j, i]

        # -- Market share evolution (Replicator dynamics, Eq 2) --
        for j in range(num_economies):
            for i in range(firms_per_economy):
                rel_fit = E_firm[j, i] / E_global
                z_firm[j, i] *= (1 + phi * (rel_fit - 1))

        # -- Update world output lags and exogenous world demand --
        Yw_prev2 = Yw_prev1.copy()
        Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
        Y_exo *= (1 + lambda_exo)

        # -- History for plotting --
        for j in range(num_economies):
            Y_hist[j].append(Y_economy[j])
            A_hist[j].append(A_economy[j])
        E_global_hist.append(E_global)
        a_global_hist.append(a_firm.mean())

    # === Plotting ===
    t = np.arange(T)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for j in range(num_economies):
        axs[0, 0].plot(t, Y_hist[j], label=f"Economy {j+1}")
        axs[0, 1].plot(t, A_hist[j])
    axs[1, 0].plot(t, E_global_hist, color="black")
    axs[1, 1].plot(t, a_global_hist, color="black")
    axs[0, 0].set_title("Y_economy (GDP, by economy)")
    axs[0, 1].set_title("A_economy (Avg. Effective Productivity)")
    axs[1, 0].set_title("E_global (Global Competitiveness)")
    axs[1, 1].set_title("a_global (Global Productivity)")
    axs[0, 0].legend()
    plt.tight_layout()
    plt.show()

    # Return results for further analysis if needed
    return {
        "Y_hist": Y_hist,
        "A_hist": A_hist,
        "E_global_hist": E_global_hist,
        "a_global_hist": a_global_hist,
        "final_Y": Y_economy,
        "final_A": A_economy,
        "final_w": w_economy,
        "final_a_firm": a_firm,
        "final_A_firm": A_firm,
        "final_z_firm": z_firm,
        "final_K_firm": K_firm
    }

def run_ll_abm_fx(
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,
    rho=0.6,
    alpha=0.375,
    beta=0.5,
    gamma=1.0,
    phi=1.0,
    lambda_exo=0.05,
    theta_innovator=0.0,
    theta_imitator=0.0,
    sigma=0.1,
    chi=0.5,
    kappa=0.5,
    ex_rate_shock_var=0.0001,
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    seed=None,
    plot=True
):  
    """
    Runs a multi-economy agent-based model (ABM) with endogenous innovation, imitation, 
    and exchange rate dynamics. Each economy consists of multiple firms competing 
    for market share, investing in productivity, and trading with other economies.

    Parameters
    ----------
    T : int, optional
        Number of time steps to simulate (default: 200).
    num_economies : int, optional
        Number of economies in the model (default: 5).
    firms_per_economy : int, optional
        Number of firms in each economy (default: 20).
    mu : float, optional
        Markup parameter for firm pricing (default: 0.6).
    iota : float, optional
        Fraction of output allocated to investment (default: 0.4).
    rho : float, optional
        Fraction of output allocated to R&D (default: 0.6).
    alpha : float, optional
        Elasticity of exports with respect to world demand (default: 0.375).
    beta : float, optional
        Scaling parameter for output update (default: 0.5).
    gamma : float, optional
        Wage update responsiveness to productivity growth (default: 1.0).
    phi : float, optional
        Strength of replicator dynamics and competitiveness effects (default: 1.0).
    lambda_exo : float, optional
        Growth rate of exogenous world demand (default: 0.05).
    theta_innovator : float, optional
        Additional innovation probability for innovator firms (default: 0.0).
    theta_imitator : float, optional
        Additional innovation probability for imitator firms (default: 0.0).
    sigma : float, optional
        Standard deviation of innovation shocks (default: 0.1).
    chi : float, optional
        Imitation intensity parameter (default: 0.5).
    kappa : float, optional
        Exchange rate sensitivity to relative export growth (default: 0.5).
    ex_rate_shock_var : float, optional
        Variance of exchange rate shocks (default: 0.0001).
    Y0 : float, optional
        Initial GDP for each economy (default: 100.0).
    w0 : float, optional
        Initial wage for each economy (default: 5.0).
    Yw0 : float, optional
        Initial world GDP (default: 401.0).
    Y_exo0 : float, optional
        Initial exogenous world demand (default: 1.0).
    seed : int or None, optional
        Random seed for reproducibility (default: None).
    plot : bool, optional
        If True, plot time series results (default: True).

    Returns
    -------
    results : dict
        Dictionary containing time series and final state of key variables:
            - "Y_hist": List of GDP time series for each economy.
            - "A_hist": List of average productivity time series for each economy.
            - "E_global_hist": List of global competitiveness over time.
            - "a_global_hist": List of global average productivity over time.
            - "ex_rate_hist": List of exchange rate time series for each economy.
            - "exports_hist": List of export time series for each economy.
            - "final_Y": Final GDP values for each economy.
            - "final_A": Final average productivity for each economy.
            - "final_w": Final wage for each economy.
            - "final_a_firm": Final productivity for each firm.
            - "final_A_firm": Final effective productivity for each firm.
            - "final_z_firm": Final market share for each firm.
            - "final_K_firm": Final capital for each firm.
            - "final_ex_rate": Final exchange rate for each economy.
            - "final_exports": Final exports for each economy.

    Notes
    -----
    - The model implements endogenous innovation and imitation at the firm level, 
      with market share dynamics governed by replicator equations.
    - Exchange rates evolve endogenously based on relative export performance and 
      stochastic shocks.
    - If `plot` is True, several key time series are visualized using matplotlib.
    """
    if seed is not None:
        np.random.seed(seed)

    # === INITIALIZATION ===
    a_firm = np.ones((num_economies, firms_per_economy))
    A_firm = np.ones((num_economies, firms_per_economy))
    z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
    K_firm = np.ones((num_economies, firms_per_economy))

    Y_firm = np.ones((num_economies, firms_per_economy))
    Pi_firm = np.ones((num_economies, firms_per_economy))
    I_firm = np.zeros((num_economies, firms_per_economy))
    R_firm = np.zeros((num_economies, firms_per_economy))
    innovators = (np.arange(firms_per_economy) % 2 == 0)

    Y_economy = np.full(num_economies, Y0)
    A_economy = np.ones(num_economies)
    E_economy = np.ones(num_economies)
    w_economy = np.full(num_economies, w0)

    Yw_prev1 = np.full(num_economies, Yw0)
    Yw_prev2 = np.full(num_economies, Yw0)
    Y_exo = Y_exo0

    ex_rate = np.ones(num_economies)
    exports = np.ones(num_economies)
    exports_prev = np.ones(num_economies)
    exports_global = exports.sum()
    exports_global_prev = exports_global

    # === STORAGE ===
    Y_hist = [[] for _ in range(num_economies)]
    A_hist = [[] for _ in range(num_economies)]
    E_global_hist = []
    a_global_hist = []
    ex_rate_hist = [[] for _ in range(num_economies)]
    exports_hist = [[] for _ in range(num_economies)]

    # === SIMULATION LOOP ===
    for t in range(T):
        z_economy = z_firm.sum(axis=1)
        A_prev = A_economy.copy()
        # Effective productivity (weighted)
        A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

        # --- STEP 1: Calculate exports for each economy ---
        exports_prev[:] = exports[:]
        for j in range(num_economies):
            Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
            exports[j] = (Yw_t ** alpha) * z_economy[j]
            exports_hist[j].append(exports[j])

        # --- STEP 2: Exchange Rate Update (relative to global export growth) ---
        exports_global_prev = exports_global
        exports_global = exports.sum()
        global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

        for j in range(num_economies):
            own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
            rel_growth = own_growth - global_growth
            shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
            ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock
            ex_rate_hist[j].append(ex_rate[j])

        # --- STEP 3: Competitiveness ---
        price_firm = (1 + mu) * w_economy[:, None] / A_firm
        E_firm = 1 / (price_firm * ex_rate[:, None])
        E_economy = (z_firm * E_firm).sum(axis=1) / z_economy

        # --- STEP 4: Wage update ---
        growth_A = (A_economy / A_prev - 1)
        w_economy *= (1 + gamma * growth_A)

        # --- STEP 5: Output Update (GDP) ---
        rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
        e_term = 1.0 / (1 - z_economy)
        E_global = (E_economy * z_economy).sum()
        phi_term = phi / beta * e_term * (E_economy / E_global - 1)
        alpha_term = alpha / beta * rel_Yw
        Y_economy *= (1 + alpha_term + phi_term)

        # --- STEP 6: Firm-level updates ---
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

        # --- STEP 7: Weighted productivity, capital update ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                denom = K_firm[j, i] + I_firm[j, i]
                if denom > 0:
                    A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                K_firm[j, i] += I_firm[j, i]

        # --- STEP 8: Market share update (Replicator dynamics) ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                rel_fit = E_firm[j, i] / E_global
                z_firm[j, i] *= (1 + phi * (rel_fit - 1))

        # --- STEP 9: Update rest-of-world GDP and exogenous world demand ---
        Yw_prev2 = Yw_prev1.copy()
        Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
        Y_exo *= (1 + lambda_exo)

        # --- STEP 10: Store time series for analysis/plotting ---
        for j in range(num_economies):
            Y_hist[j].append(Y_economy[j])
            A_hist[j].append(A_economy[j])
        E_global_hist.append(E_global)
        a_global_hist.append(a_firm.mean())

    # === PLOTTING ===
    if plot:
        t = np.arange(T)
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        for j in range(num_economies):
            axs[0, 0].plot(t, Y_hist[j], label=f"Economy {j+1}")
            axs[0, 1].plot(t, A_hist[j])
            axs[1, 0].plot(t, ex_rate_hist[j])
            axs[1, 1].plot(t, exports_hist[j])
        axs[2, 0].plot(t, E_global_hist, color="black")
        axs[2, 1].plot(t, a_global_hist, color="black")
        axs[0, 0].set_title("Y_economy (GDP, by economy)")
        axs[0, 1].set_title("A_economy (Avg. Effective Productivity)")
        axs[1, 0].set_title("Exchange Rate by Economy")
        axs[1, 1].set_title("Exports by Economy")
        axs[2, 0].set_title("E_global (Global Competitiveness)")
        axs[2, 1].set_title("a_global (Global Productivity)")
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()

    # Return results for further analysis if needed
    return {
        "Y_hist": Y_hist,
        "A_hist": A_hist,
        "E_global_hist": E_global_hist,
        "a_global_hist": a_global_hist,
        "ex_rate_hist": ex_rate_hist,
        "exports_hist": exports_hist,
        "final_Y": Y_economy,
        "final_A": A_economy,
        "final_w": w_economy,
        "final_a_firm": a_firm,
        "final_A_firm": A_firm,
        "final_z_firm": z_firm,
        "final_K_firm": K_firm,
        "final_ex_rate": ex_rate,
        "final_exports": exports
    }

def run_ll_abm_fx_with_depreciation(
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,
    rho=0.6,
    alpha=0.375,
    beta=0.5,
    gamma=1.0,
    phi=1.0,
    lambda_exo=0.05,
    theta_innovator=0.0,
    theta_imitator=0.0,
    sigma=0.1,
    chi=0.5,
    kappa=0.5,
    ex_rate_shock_var=0.0001,
    delta_K=0.05,
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    seed=None,
    plot=True
):
    """
    Simulates a multi-economy agent-based model (ABM) with endogenous exchange rates, 
    firm-level innovation and imitation, capital depreciation, and international trade.

    Parameters
    ----------
    T : int, optional
        Number of time steps to simulate (default: 200).
    num_economies : int, optional
        Number of economies in the model (default: 5).
    firms_per_economy : int, optional
        Number of firms in each economy (default: 20).
    mu : float, optional
        Markup parameter for firm pricing (default: 0.6).
    iota : float, optional
        Fraction of output invested by firms (default: 0.4).
    rho : float, optional
        Fraction of output allocated to R&D (default: 0.6).
    alpha : float, optional
        Exports elasticity parameter (default: 0.375).
    beta : float, optional
        Scaling parameter for output update (default: 0.5).
    gamma : float, optional
        Wage growth sensitivity to productivity (default: 1.0).
    phi : float, optional
        Selection strength in market share update (default: 1.0).
    lambda_exo : float, optional
        Growth rate of exogenous world demand (default: 0.05).
    theta_innovator : float, optional
        Innovation success boost for innovators (default: 0.0).
    theta_imitator : float, optional
        Innovation success boost for imitators (default: 0.0).
    sigma : float, optional
        Standard deviation of innovation shocks (default: 0.1).
    chi : float, optional
        Imitation gap scaling parameter (default: 0.5).
    kappa : float, optional
        Exchange rate adjustment sensitivity (default: 0.5).
    ex_rate_shock_var : float, optional
        Variance of exchange rate shocks (default: 0.0001).
    delta_K : float, optional
        Capital depreciation rate (default: 0.05).
    Y0 : float, optional
        Initial GDP for each economy (default: 100.0).
    w0 : float, optional
        Initial wage for each economy (default: 5.0).
    Yw0 : float, optional
        Initial world GDP (default: 401.0).
    Y_exo0 : float, optional
        Initial exogenous world demand (default: 1.0).
    seed : int or None, optional
        Random seed for reproducibility (default: None).
    plot : bool, optional
        If True, plot simulation results (default: True).

    Returns
    -------
    dict
        Dictionary containing time series and final states:
            - "Y_hist": List of GDP time series for each economy.
            - "A_hist": List of average productivity time series for each economy.
            - "E_global_hist": List of global competitiveness over time.
            - "a_global_hist": List of global average productivity over time.
            - "ex_rate_hist": List of exchange rate time series for each economy.
            - "exports_hist": List of exports time series for each economy.
            - "final_Y": Final GDP values for each economy.
            - "final_A": Final average productivity for each economy.
            - "final_w": Final wage values for each economy.
            - "final_a_firm": Final firm-level productivity array.
            - "final_A_firm": Final firm-level effective productivity array.
            - "final_z_firm": Final firm-level market shares.
            - "final_K_firm": Final firm-level capital stocks.
            - "final_ex_rate": Final exchange rates for each economy.
            - "final_exports": Final exports for each economy.

    Notes
    -----
    - The model features endogenous firm-level innovation and imitation, capital accumulation and depreciation, 
      and replicator dynamics for market share evolution.
    - Exchange rates evolve endogenously based on relative export growth and random shocks.
    - Optionally, the function plots key macroeconomic and microeconomic time series.
    """
    if seed is not None:
        np.random.seed(seed)

    # === INITIALIZATION ===
    a_firm = np.ones((num_economies, firms_per_economy))
    A_firm = np.ones((num_economies, firms_per_economy))
    z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
    K_firm = np.ones((num_economies, firms_per_economy))

    Y_firm = np.ones((num_economies, firms_per_economy))
    Pi_firm = np.ones((num_economies, firms_per_economy))
    I_firm = np.zeros((num_economies, firms_per_economy))
    R_firm = np.zeros((num_economies, firms_per_economy))
    innovators = (np.arange(firms_per_economy) % 2 == 0)

    Y_economy = np.full(num_economies, Y0)
    A_economy = np.ones(num_economies)
    E_economy = np.ones(num_economies)
    w_economy = np.full(num_economies, w0)

    Yw_prev1 = np.full(num_economies, Yw0)
    Yw_prev2 = np.full(num_economies, Yw0)
    Y_exo = Y_exo0

    ex_rate = np.ones(num_economies)
    exports = np.ones(num_economies)
    exports_prev = np.ones(num_economies)
    exports_global = exports.sum()
    exports_global_prev = exports_global

    # === STORAGE ===
    Y_hist = [[] for _ in range(num_economies)]
    A_hist = [[] for _ in range(num_economies)]
    E_global_hist = []
    a_global_hist = []
    ex_rate_hist = [[] for _ in range(num_economies)]
    exports_hist = [[] for _ in range(num_economies)]

    # === SIMULATION LOOP ===
    for t in range(T):
        z_economy = z_firm.sum(axis=1)
        A_prev = A_economy.copy()
        A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

        # --- STEP 1: Calculate exports for each economy ---
        exports_prev[:] = exports[:]
        for j in range(num_economies):
            Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
            exports[j] = (Yw_t ** alpha) * z_economy[j]
            exports_hist[j].append(exports[j])

        # --- STEP 2: Exchange Rate Update (relative to global export growth) ---
        exports_global_prev = exports_global
        exports_global = exports.sum()
        global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

        for j in range(num_economies):
            own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
            rel_growth = own_growth - global_growth
            shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
            ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock
            ex_rate_hist[j].append(ex_rate[j])

        # --- STEP 3: Competitiveness ---
        price_firm = (1 + mu) * w_economy[:, None] / A_firm
        E_firm = 1 / (price_firm * ex_rate[:, None])
        E_economy = (z_firm * E_firm).sum(axis=1) / z_economy

        # --- STEP 4: Wage update ---
        growth_A = (A_economy / A_prev - 1)
        w_economy *= (1 + gamma * growth_A)

        # --- STEP 5: Output Update (GDP) ---
        rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
        e_term = 1.0 / (1 - z_economy)
        E_global = (E_economy * z_economy).sum()
        phi_term = phi / beta * e_term * (E_economy / E_global - 1)
        alpha_term = alpha / beta * rel_Yw
        Y_economy *= (1 + alpha_term + phi_term)

        # --- STEP 6: Firm-level updates ---
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

        # --- STEP 7: Weighted productivity, capital depreciation, capital update ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                denom = K_firm[j, i] + I_firm[j, i]
                if denom > 0:
                    A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                # Apply capital depreciation and add new investment
                K_firm[j, i] = (1 - delta_K) * K_firm[j, i] + I_firm[j, i]

        # --- STEP 8: Market share update (Replicator dynamics) ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                rel_fit = E_firm[j, i] / E_global
                z_firm[j, i] *= (1 + phi * (rel_fit - 1))

        # --- STEP 9: Update rest-of-world GDP and exogenous world demand ---
        Yw_prev2 = Yw_prev1.copy()
        Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
        Y_exo *= (1 + lambda_exo)

        # --- STEP 10: Store time series for analysis/plotting ---
        for j in range(num_economies):
            Y_hist[j].append(Y_economy[j])
            A_hist[j].append(A_economy[j])
        E_global_hist.append(E_global)
        a_global_hist.append(a_firm.mean())

    # === PLOTTING ===
    if plot:
        t = np.arange(T)
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        for j in range(num_economies):
            axs[0, 0].plot(t, Y_hist[j], label=f"Economy {j+1}")
            axs[0, 1].plot(t, A_hist[j])
            axs[1, 0].plot(t, ex_rate_hist[j])
            axs[1, 1].plot(t, exports_hist[j])
        axs[2, 0].plot(t, E_global_hist, color="black")
        axs[2, 1].plot(t, a_global_hist, color="black")
        axs[0, 0].set_title("Y_economy (GDP, by economy)")
        axs[0, 1].set_title("A_economy (Avg. Effective Productivity)")
        axs[1, 0].set_title("Exchange Rate by Economy")
        axs[1, 1].set_title("Exports by Economy")
        axs[2, 0].set_title("E_global (Global Competitiveness)")
        axs[2, 1].set_title("a_global (Global Productivity)")
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()

    return {
        "Y_hist": Y_hist,
        "A_hist": A_hist,
        "E_global_hist": E_global_hist,
        "a_global_hist": a_global_hist,
        "ex_rate_hist": ex_rate_hist,
        "exports_hist": exports_hist,
        "final_Y": Y_economy,
        "final_A": A_economy,
        "final_w": w_economy,
        "final_a_firm": a_firm,
        "final_A_firm": A_firm,
        "final_z_firm": z_firm,
        "final_K_firm": K_firm,
        "final_ex_rate": ex_rate,
        "final_exports": exports
    }

def run_ll_abm_fx_with_tariff(
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,
    rho=0.6,
    alpha=0.375,
    beta=0.5,
    gamma=1.0,
    phi=1.0,
    lambda_exo=0.05,
    theta_innovator=0.0,
    theta_imitator=0.0,
    sigma=0.1,
    chi=0.5,
    kappa=0.5,
    ex_rate_shock_var=0.0001,
    delta_K=0.05,
    tariff_rate=0.10,
    tariff_economy=0,
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    seed=None,
    plot=True
):
    """
        Simulates a multi-economy agent-based model (ABM) with endogenous exchange rates, firm-level innovation,
        and the option to impose a tariff on one economy. The model tracks the evolution of productivity,
        competitiveness, exchange rates, and output (GDP) across economies and firms over time.

        Parameters
        ----------
        T : int, optional
            Number of time steps to simulate (default: 200).
        num_economies : int, optional
            Number of economies in the simulation (default: 5).
        firms_per_economy : int, optional
            Number of firms per economy (default: 20).
        mu : float, optional
            Markup rate for firm pricing (default: 0.6).
        iota : float, optional
            Fraction of output reinvested as investment by firms (default: 0.4).
        rho : float, optional
            Fraction of output allocated to R&D by firms (default: 0.6).
        alpha : float, optional
            Elasticity parameter for exports (default: 0.375).
        beta : float, optional
            Parameter for output update (default: 0.5).
        gamma : float, optional
            Wage growth sensitivity to productivity growth (default: 1.0).
        phi : float, optional
            Selection strength in replicator dynamics (default: 1.0).
        lambda_exo : float, optional
            Growth rate of exogenous world demand (default: 0.05).
        theta_innovator : float, optional
            Additional innovation probability for innovator firms (default: 0.0).
        theta_imitator : float, optional
            Additional innovation probability for imitator firms (default: 0.0).
        sigma : float, optional
            Standard deviation of innovation shock for innovators (default: 0.1).
        chi : float, optional
            Imitation intensity parameter (default: 0.5).
        kappa : float, optional
            Exchange rate sensitivity to relative export growth (default: 0.5).
        ex_rate_shock_var : float, optional
            Variance of exchange rate shocks (default: 0.0001).
        delta_K : float, optional
            Capital depreciation rate (default: 0.05).
        tariff_rate : float, optional
            Tariff rate applied to imports into the specified economy (default: 0.10).
        tariff_economy : int, optional
            Index of the economy imposing the tariff (default: 0).
        Y0 : float, optional
            Initial GDP for each economy (default: 100.0).
        w0 : float, optional
            Initial wage for each economy (default: 5.0).
        Yw0 : float, optional
            Initial rest-of-world GDP for each economy (default: 401.0).
        Y_exo0 : float, optional
            Initial exogenous world demand (default: 1.0).
        seed : int or None, optional
            Random seed for reproducibility (default: None).
        plot : bool, optional
            If True, plot time series of key variables (default: True).

        Returns
        -------
        dict
            Dictionary containing time series and final values for key variables:
                - "Y_hist": List of GDP time series for each economy.
                - "A_hist": List of average productivity time series for each economy.
                - "E_global_hist": List of global competitiveness over time.
                - "a_global_hist": List of global average productivity over time.
                - "ex_rate_hist": List of exchange rate time series for each economy.
                - "exports_hist": List of export time series for each economy.
                - "final_Y": Final GDP values for each economy.
                - "final_A": Final average productivity for each economy.
                - "final_w": Final wage for each economy.
                - "final_a_firm": Final firm-level productivity array.
                - "final_A_firm": Final firm-level effective productivity array.
                - "final_z_firm": Final firm-level market shares.
                - "final_K_firm": Final firm-level capital stocks.
                - "final_ex_rate": Final exchange rates for each economy.
                - "final_exports": Final exports for each economy.

        Notes
        -----
        - The model implements firm-level innovation and imitation, endogenous exchange rates, and replicator dynamics for market shares.
        - Tariffs are applied to imports into the specified economy, affecting competitiveness and trade flows.
        - Plots are generated if `plot=True`, showing the evolution of key macroeconomic and microeconomic variables. 
        """
    
    if seed is not None:
        np.random.seed(seed)

    print(f"Tariffs are imposed by Economy {tariff_economy + 1} (index {tariff_economy}) at a rate of {100 * tariff_rate:.1f}% on all imports from other economies.")

    # === INITIALIZATION ===
    a_firm = np.ones((num_economies, firms_per_economy))
    A_firm = np.ones((num_economies, firms_per_economy))
    z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
    K_firm = np.ones((num_economies, firms_per_economy))

    Y_firm = np.ones((num_economies, firms_per_economy))
    Pi_firm = np.ones((num_economies, firms_per_economy))
    I_firm = np.zeros((num_economies, firms_per_economy))
    R_firm = np.zeros((num_economies, firms_per_economy))
    innovators = (np.arange(firms_per_economy) % 2 == 0)

    Y_economy = np.full(num_economies, Y0)
    A_economy = np.ones(num_economies)
    E_economy = np.ones(num_economies)
    w_economy = np.full(num_economies, w0)

    Yw_prev1 = np.full(num_economies, Yw0)
    Yw_prev2 = np.full(num_economies, Yw0)
    Y_exo = Y_exo0

    ex_rate = np.ones(num_economies)
    exports = np.ones(num_economies)
    exports_prev = np.ones(num_economies)
    exports_global = exports.sum()
    exports_global_prev = exports_global

    # === STORAGE ===
    Y_hist = [[] for _ in range(num_economies)]
    A_hist = [[] for _ in range(num_economies)]
    E_global_hist = []
    a_global_hist = []
    ex_rate_hist = [[] for _ in range(num_economies)]
    exports_hist = [[] for _ in range(num_economies)]

    # === SIMULATION LOOP ===
    for t in range(T):
        z_economy = z_firm.sum(axis=1)
        A_prev = A_economy.copy()
        A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

        # --- STEP 1: Calculate exports for each economy ---
        exports_prev[:] = exports[:]
        for j in range(num_economies):
            Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
            exports[j] = (Yw_t ** alpha) * z_economy[j]
            exports_hist[j].append(exports[j])

        # --- STEP 2: Exchange Rate Update (relative to global export growth) ---
        exports_global_prev = exports_global
        exports_global = exports.sum()
        global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

        for j in range(num_economies):
            own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
            rel_growth = own_growth - global_growth
            shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
            ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock
            ex_rate_hist[j].append(ex_rate[j])

        # --- STEP 3: Competitiveness with Tariff ---
        price_firm = (1 + mu) * w_economy[:, None] / A_firm
        E_firm = np.zeros((num_economies, firms_per_economy))
        for j in range(num_economies):  # For each economy
            for i in range(firms_per_economy):  # For each firm in economy j
                effective_price = price_firm[j, i] * ex_rate[j]
                # Apply tariff to firms not from tariff_economy when exporting to tariff_economy
                firm_origin = i // firms_per_economy
                if j == tariff_economy and j != firm_origin:
                    effective_price *= (1 + tariff_rate)
                E_firm[j, i] = 1 / effective_price
        E_economy = (z_firm * E_firm).sum(axis=1) / z_economy

        # --- STEP 4: Wage update ---
        growth_A = (A_economy / A_prev - 1)
        w_economy *= (1 + gamma * growth_A)

        # --- STEP 5: Output Update (GDP) ---
        rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
        e_term = 1.0 / (1 - z_economy)
        E_global = (E_economy * z_economy).sum()
        phi_term = phi / beta * e_term * (E_economy / E_global - 1)
        alpha_term = alpha / beta * rel_Yw
        Y_economy *= (1 + alpha_term + phi_term)

        # --- STEP 6: Firm-level updates ---
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

        # --- STEP 7: Weighted productivity, capital depreciation, capital update ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                denom = K_firm[j, i] + I_firm[j, i]
                if denom > 0:
                    A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                K_firm[j, i] = (1 - delta_K) * K_firm[j, i] + I_firm[j, i]

        # --- STEP 8: Market share update (Replicator dynamics) ---
        for j in range(num_economies):
            for i in range(firms_per_economy):
                rel_fit = E_firm[j, i] / E_global
                z_firm[j, i] *= (1 + phi * (rel_fit - 1))

        # --- STEP 9: Update rest-of-world GDP and exogenous world demand ---
        Yw_prev2 = Yw_prev1.copy()
        Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
        Y_exo *= (1 + lambda_exo)

        # --- STEP 10: Store time series for analysis/plotting ---
        for j in range(num_economies):
            Y_hist[j].append(Y_economy[j])
            A_hist[j].append(A_economy[j])
        E_global_hist.append(E_global)
        a_global_hist.append(a_firm.mean())

    # === PLOTTING ===
    if plot:
        t = np.arange(T)
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        for j in range(num_economies):
            axs[0, 0].plot(t, Y_hist[j], label=f"Economy {j+1}")
            axs[0, 1].plot(t, A_hist[j])
            axs[1, 0].plot(t, ex_rate_hist[j])
            axs[1, 1].plot(t, exports_hist[j])
        axs[2, 0].plot(t, E_global_hist, color="black")
        axs[2, 1].plot(t, a_global_hist, color="black")
        axs[0, 0].set_title("Y_economy (GDP, by economy)")
        axs[0, 1].set_title("A_economy (Avg. Effective Productivity)")
        axs[1, 0].set_title("Exchange Rate by Economy")
        axs[1, 1].set_title("Exports by Economy")
        axs[2, 0].set_title("E_global (Global Competitiveness)")
        axs[2, 1].set_title("a_global (Global Productivity)")
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()

    return {
        "Y_hist": Y_hist,
        "A_hist": A_hist,
        "E_global_hist": E_global_hist,
        "a_global_hist": a_global_hist,
        "ex_rate_hist": ex_rate_hist,
        "exports_hist": exports_hist,
        "final_Y": Y_economy,
        "final_A": A_economy,
        "final_w": w_economy,
        "final_a_firm": a_firm,
        "final_A_firm": A_firm,
        "final_z_firm": z_firm,
        "final_K_firm": K_firm,
        "final_ex_rate": ex_rate,
        "final_exports": exports
    }

def run_ll_abm_fx_with_dynamic_tariff(
        T=100,
        num_economies=5,
        firms_per_economy=20,
        mu=0.6,
        iota=0.4,
        rho=0.6,
        alpha=0.375,
        beta=0.5,
        gamma=1.0,
        phi=1.0,
        lambda_exo=0.05,
        theta_innovator=0.0,
        theta_imitator=0.0,
        sigma=0.1,
        chi=0.5,
        kappa=0.4,
        ex_rate_shock_var=0.0001,
        delta_K=0.05,
        tariff_rate=0.10,
        tariff_start=60,
        Y0=100.0,
        w0=5.0,
        Yw0=401.0,
        Y_exo0=1.0,
        seed=None,
        plot=True
    ):        
        """
        Simulates a multi-economy agent-based model (ABM) with endogenous exchange rates, firm-level productivity dynamics,
        and a dynamic tariff regime imposed by the leading economy after a specified period.

        The model features:
        - Multiple economies, each with multiple firms.
        - Firm-level innovation and imitation processes affecting productivity.
        - Endogenous exchange rates based on relative export growth.
        - Dynamic tariffs imposed by the leading economy after `tariff_start` periods.
        - Replicator dynamics for firm market shares.
        - Capital accumulation and depreciation at the firm level.
        - Endogenous wage and output (GDP) updates.
        - Optional plotting of key time series.

        Parameters
        ----------
        T : int, optional
            Number of simulation periods (default: 100).
        num_economies : int, optional
            Number of economies in the simulation (default: 5).
        firms_per_economy : int, optional
            Number of firms per economy (default: 20).
        mu : float, optional
            Markup parameter for firm pricing (default: 0.6).
        iota : float, optional
            Fraction of output invested by firms (default: 0.4).
        rho : float, optional
            Fraction of output allocated to R&D by firms (default: 0.6).
        alpha : float, optional
            Output elasticity parameter (default: 0.375).
        beta : float, optional
            Output elasticity parameter (default: 0.5).
        gamma : float, optional
            Wage update parameter (default: 1.0).
        phi : float, optional
            Replicator dynamics parameter (default: 1.0).
        lambda_exo : float, optional
            Growth rate of exogenous world demand (default: 0.05).
        theta_innovator : float, optional
            Innovation probability boost for innovators (default: 0.0).
        theta_imitator : float, optional
            Innovation probability boost for imitators (default: 0.0).
        sigma : float, optional
            Standard deviation of innovation shocks (default: 0.1).
        chi : float, optional
            Imitation gap scaling parameter (default: 0.5).
        kappa : float, optional
            Exchange rate sensitivity to relative export growth (default: 0.4).
        ex_rate_shock_var : float, optional
            Variance of exchange rate shocks (default: 0.0001).
        delta_K : float, optional
            Capital depreciation rate (default: 0.05).
        tariff_rate : float, optional
            Tariff rate imposed by the leading economy (default: 0.10).
        tariff_start : int, optional
            Period after which the leading economy imposes tariffs (default: 60).
        Y0 : float, optional
            Initial GDP for each economy (default: 100.0).
        w0 : float, optional
            Initial wage for each economy (default: 5.0).
        Yw0 : float, optional
            Initial rest-of-world GDP (default: 401.0).
        Y_exo0 : float, optional
            Initial exogenous world demand (default: 1.0).
        seed : int or None, optional
            Random seed for reproducibility (default: None).
        plot : bool, optional
            If True, plot key time series at the end of the simulation (default: True).

        Returns
        -------
        dict
            Dictionary containing time series and final states:
                - "Y_hist": List of GDP time series for each economy.
                - "A_hist": List of average productivity time series for each economy.
                - "E_global_hist": List of global competitiveness over time.
                - "a_global_hist": List of global average productivity over time.
                - "ex_rate_hist": List of exchange rate time series for each economy.
                - "exports_hist": List of export time series for each economy.
                - "final_Y": Final GDP values for each economy.
                - "final_A": Final average productivity for each economy.
                - "final_w": Final wage for each economy.
                - "final_a_firm": Final firm-level productivity array.
                - "final_A_firm": Final firm-level effective productivity array.
                - "final_z_firm": Final firm-level market shares.
                - "final_K_firm": Final firm-level capital stocks.
                - "final_ex_rate": Final exchange rates for each economy.
                - "final_exports": Final exports for each economy.
                - "tariff_economy": Index of the economy that imposed tariffs (or None if not triggered).
        """

        if seed is not None:
            np.random.seed(seed)

        # === INITIALIZATION ===
        a_firm = np.ones((num_economies, firms_per_economy))
        A_firm = np.ones((num_economies, firms_per_economy))
        z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
        K_firm = np.ones((num_economies, firms_per_economy))

        Y_firm = np.ones((num_economies, firms_per_economy))
        Pi_firm = np.ones((num_economies, firms_per_economy))
        I_firm = np.zeros((num_economies, firms_per_economy))
        R_firm = np.zeros((num_economies, firms_per_economy))
        innovators = (np.arange(firms_per_economy) % 2 == 0)

        Y_economy = np.full(num_economies, Y0)
        A_economy = np.ones(num_economies)
        E_economy = np.ones(num_economies)
        w_economy = np.full(num_economies, w0)

        Yw_prev1 = np.full(num_economies, Yw0)
        Yw_prev2 = np.full(num_economies, Yw0)
        Y_exo = Y_exo0

        ex_rate = np.ones(num_economies)
        exports = np.ones(num_economies)
        exports_prev = np.ones(num_economies)
        exports_global = exports.sum()
        exports_global_prev = exports_global

        # === STORAGE ===
        Y_hist = [[] for _ in range(num_economies)]
        A_hist = [[] for _ in range(num_economies)]
        E_global_hist = []
        a_global_hist = []
        ex_rate_hist = [[] for _ in range(num_economies)]
        exports_hist = [[] for _ in range(num_economies)]

        tariff_economy = None

        # === SIMULATION LOOP ===
        for t in range(T):
            z_economy = z_firm.sum(axis=1)
            A_prev = A_economy.copy()
            A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

            # --- STEP 1: Calculate exports for each economy ---
            exports_prev[:] = exports[:]
            for j in range(num_economies):
                Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
                exports[j] = (Yw_t ** alpha) * z_economy[j]
                exports_hist[j].append(exports[j])

            # --- STEP 2: Exchange Rate Update (relative to global export growth) ---
            exports_global_prev = exports_global
            exports_global = exports.sum()
            global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

            for j in range(num_economies):
                own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
                rel_growth = own_growth - global_growth
                shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
                ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock
                ex_rate_hist[j].append(ex_rate[j])

            # --- Tariff regime activates at t=tariff_start (after period 60) ---
            if t == tariff_start:
                last_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
                tariff_economy = np.argmax(last_gdp)
                print(f"At turn {t}, Economy {tariff_economy+1} is the leader and imposes tariffs.")

            # --- STEP 3: Competitiveness with dynamic Tariff (if imposed) ---
            price_firm = (1 + mu) * w_economy[:, None] / A_firm
            E_firm = np.zeros((num_economies, firms_per_economy))
            for j in range(num_economies):  # For each economy
                for i in range(firms_per_economy):  # For each firm in economy j
                    effective_price = price_firm[j, i] * ex_rate[j]
                    # If tariffs have started, and exporting to tariff_economy, and not from there
                    if tariff_economy is not None and j == tariff_economy and j != i // firms_per_economy:
                        effective_price *= (1 + tariff_rate)
                    E_firm[j, i] = 1 / effective_price
            E_economy = (z_firm * E_firm).sum(axis=1) / z_economy

            # --- STEP 4: Wage update ---
            growth_A = (A_economy / A_prev - 1)
            w_economy *= (1 + gamma * growth_A)

            # --- STEP 5: Output Update (GDP) ---
            rel_Yw = (Yw_prev1 / Yw_prev2 - 1)
            e_term = 1.0 / (1 - z_economy)
            E_global = (E_economy * z_economy).sum()
            phi_term = phi / beta * e_term * (E_economy / E_global - 1)
            alpha_term = alpha / beta * rel_Yw
            Y_economy *= (1 + alpha_term + phi_term)

            # --- STEP 6: Firm-level updates ---
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

            # --- STEP 7: Weighted productivity, capital depreciation, capital update ---
            for j in range(num_economies):
                for i in range(firms_per_economy):
                    denom = K_firm[j, i] + I_firm[j, i]
                    if denom > 0:
                        A_firm[j, i] = I_firm[j, i] / denom * a_firm[j, i] + K_firm[j, i] / denom * A_firm[j, i]
                    K_firm[j, i] = (1 - delta_K) * K_firm[j, i] + I_firm[j, i]

            # --- STEP 8: Market share update (Replicator dynamics) ---
            for j in range(num_economies):
                for i in range(firms_per_economy):
                    rel_fit = E_firm[j, i] / E_global
                    z_firm[j, i] *= (1 + phi * (rel_fit - 1))

            # --- STEP 9: Update rest-of-world GDP and exogenous world demand ---
            Yw_prev2 = Yw_prev1.copy()
            Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
            Y_exo *= (1 + lambda_exo)

            # --- STEP 10: Store time series for analysis/plotting ---
            for j in range(num_economies):
                Y_hist[j].append(Y_economy[j])
                A_hist[j].append(A_economy[j])
            E_global_hist.append(E_global)
            a_global_hist.append(a_firm.mean())

        # === PLOTTING ===
        if plot:
            t = np.arange(T)
            fig, axs = plt.subplots(3, 2, figsize=(15, 12))
            for j in range(num_economies):
                axs[0, 0].plot(t, Y_hist[j], label=f"Economy {j+1}")
                axs[0, 1].plot(t, A_hist[j])
                axs[1, 0].plot(t, ex_rate_hist[j])
                axs[1, 1].plot(t, exports_hist[j])
            axs[2, 0].plot(t, E_global_hist, color="black")
            axs[2, 1].plot(t, a_global_hist, color="black")
            axs[0, 0].set_title("Y_economy (GDP, by economy)")
            axs[0, 1].set_title("A_economy (Avg. Effective Productivity)")
            axs[1, 0].set_title("Exchange Rate by Economy")
            axs[1, 1].set_title("Exports by Economy")
            axs[2, 0].set_title("E_global (Global Competitiveness)")
            axs[2, 1].set_title("a_global (Global Productivity)")
            axs[0, 0].legend()
            plt.tight_layout()
            plt.show()

        return {
            "Y_hist": Y_hist,
            "A_hist": A_hist,
            "E_global_hist": E_global_hist,
            "a_global_hist": a_global_hist,
            "ex_rate_hist": ex_rate_hist,
            "exports_hist": exports_hist,
            "final_Y": Y_economy,
            "final_A": A_economy,
            "final_w": w_economy,
            "final_a_firm": a_firm,
            "final_A_firm": A_firm,
            "final_z_firm": z_firm,
            "final_K_firm": K_firm,
            "final_ex_rate": ex_rate,
            "final_exports": exports,
            "tariff_economy": tariff_economy
        }

def simulate_leadership_statistics_no_tariff(
    num_runs=1000,
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,
    rho=0.6,
    alpha=0.375,
    beta=0.5,
    gamma=1.0,
    phi=1.0,
    lambda_exo=0.05,
    theta_innovator=0.0,
    theta_imitator=0.0,
    sigma=0.1,
    chi=0.5,
    kappa=0.5,
    ex_rate_shock_var=0.0001,
    delta_K=0.05,
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    leader_check_time=60,
    verbose=False
):
    """
    Runs repeated simulations of an agent-based model (ABM) of multiple economies without tariffs, tracking leadership and divergence statistics.
    Parameters
    ----------
    num_runs : int, optional
        Number of independent simulation runs to perform (default: 1000).
    T : int, optional
        Number of time steps per simulation run (default: 100).
    num_economies : int, optional
        Number of economies in the simulation (default: 5).
    firms_per_economy : int, optional
        Number of firms per economy (default: 20).
    mu : float, optional
        Markup parameter for firm pricing (default: 0.6).
    iota : float, optional
        Fraction of output invested by firms (default: 0.4).
    rho : float, optional
        Fraction of output allocated to R&D by firms (default: 0.6).
    alpha : float, optional
        Elasticity parameter for exports (default: 0.375).
    beta : float, optional
        Parameter for economic adjustment (default: 0.5).
    gamma : float, optional
        Wage adjustment parameter (default: 1.0).
    phi : float, optional
        Selection strength parameter (default: 1.0).
    lambda_exo : float, optional
        Exogenous growth rate of external demand (default: 0.05).
    theta_innovator : float, optional
        Innovation probability boost for innovator firms (default: 0.0).
    theta_imitator : float, optional
        Innovation probability boost for imitator firms (default: 0.0).
    sigma : float, optional
        Standard deviation of innovation shocks (default: 0.1).
    chi : float, optional
        Strength of imitation (default: 0.5).
    kappa : float, optional
        Exchange rate adjustment parameter (default: 0.5).
    ex_rate_shock_var : float, optional
        Variance of exchange rate shocks (default: 0.0001).
    delta_K : float, optional
        Capital depreciation rate (default: 0.05).
    Y0 : float, optional
        Initial GDP per economy (default: 100.0).
    w0 : float, optional
        Initial wage per economy (default: 5.0).
    Yw0 : float, optional
        Initial world output (default: 401.0).
    Y_exo0 : float, optional
        Initial exogenous demand (default: 1.0).
    leader_check_time : int, optional
        Time step at which to record the current leader (default: 60).
    verbose : bool, optional
        If True, prints summary statistics after simulation (default: False).
    Returns
    -------
    dict
        Dictionary containing:
            - "fraction_persistent": Fraction of runs where the leader at `leader_check_time` remains leader at final time.
            - "who_leader_at_check": Array counting which economy was leader at `leader_check_time`.
            - "who_won": Array counting which economy was leader at final time.
            - "mean_num_switches": Mean number of leadership transitions per run.
            - "std_num_switches": Standard deviation of leadership transitions per run.
            - "gdp_gap_leader_second": List of GDP ratios (leader/second) at final time for each run.
            - "gdp_gap_leader_mean": List of GDP ratios (leader/mean of others) at final time for each run.
            - "gdp_gap_leader_last": List of GDP ratios (leader/last) at final time for each run.
            - "export_gap_leader_second": List of export ratios (leader/second) at final time for each run.
            - "export_gap_leader_mean": List of export ratios (leader/mean of others) at final time for each run.
            - "export_gap_leader_last": List of export ratios (leader/last) at final time for each run.
            - "mean_gdp_gap_leader_second": Mean GDP gap (leader/second) across runs.
            - "mean_gdp_gap_leader_mean": Mean GDP gap (leader/mean of others) across runs.
            - "mean_gdp_gap_leader_last": Mean GDP gap (leader/last) across runs.
            - "mean_export_gap_leader_second": Mean export gap (leader/second) across runs.
            - "mean_export_gap_leader_mean": Mean export gap (leader/mean of others) across runs.
            - "mean_export_gap_leader_last": Mean export gap (leader/last) across runs.
    Notes
     - The function uses numpy for all numerical operations and random number generation.
    - Each simulation run is seeded for reproducibility.
    - Leadership is defined as the economy with the highest GDP at a given time.
    - Gaps are ratios of GDP or exports between the leader and other economies.
    -----    
    """
    count_persistent = 0
    who_leader_at_check = np.zeros(num_economies, dtype=int)
    who_won = np.zeros(num_economies, dtype=int)
    num_switches_list = []
    gdp_gap_leader_second = []
    gdp_gap_leader_mean = []
    gdp_gap_leader_last = []
    export_gap_leader_second = []
    export_gap_leader_mean = []
    export_gap_leader_last = []
    
    for run in range(num_runs):
        a_firm = np.ones((num_economies, firms_per_economy))
        A_firm = np.ones((num_economies, firms_per_economy))
        z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
        K_firm = np.ones((num_economies, firms_per_economy))
        Y_firm = np.ones((num_economies, firms_per_economy))
        Pi_firm = np.ones((num_economies, firms_per_economy))
        I_firm = np.zeros((num_economies, firms_per_economy))
        R_firm = np.zeros((num_economies, firms_per_economy))
        innovators = (np.arange(firms_per_economy) % 2 == 0)
        Y_economy = np.full(num_economies, Y0)
        A_economy = np.ones(num_economies)
        E_economy = np.ones(num_economies)
        w_economy = np.full(num_economies, w0)
        Yw_prev1 = np.full(num_economies, Yw0)
        Yw_prev2 = np.full(num_economies, Yw0)
        Y_exo = Y_exo0
        ex_rate = np.ones(num_economies)
        exports = np.ones(num_economies)
        exports_prev = np.ones(num_economies)
        exports_global = exports.sum()
        exports_global_prev = exports_global

        leader_at_check = None
        np.random.seed(run)  # Reproducibility per run

        Y_hist = [[] for _ in range(num_economies)]
        Exports_hist = [[] for _ in range(num_economies)]
        leader_history = []

        for t in range(T):
            z_economy = z_firm.sum(axis=1)
            A_prev = A_economy.copy()
            A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

            exports_prev[:] = exports[:]
            for j in range(num_economies):
                Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
                exports[j] = (Yw_t ** alpha) * z_economy[j]
                Exports_hist[j].append(exports[j])

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
                    K_firm[j, i] = (1 - delta_K) * K_firm[j, i] + I_firm[j, i]

            for j in range(num_economies):
                for i in range(firms_per_economy):
                    rel_fit = E_firm[j, i] / E_global
                    z_firm[j, i] *= (1 + phi * (rel_fit - 1))

            Yw_prev2 = Yw_prev1.copy()
            Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
            Y_exo *= (1 + lambda_exo)

            for j in range(num_economies):
                Y_hist[j].append(Y_economy[j])
            # Track leader
            current_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
            leader_history.append(np.argmax(current_gdp))

            # Record leader at check time
            if t == leader_check_time:
                leader_at_check = np.argmax(current_gdp)
                who_leader_at_check[leader_at_check] += 1

        # At end: Check final winner, and statistics
        final_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
        winner = np.argmax(final_gdp)
        who_won[winner] += 1
        if winner == leader_at_check:
            count_persistent += 1

        # Leadership switches (count transitions in leader_history)
        num_switches = np.sum(np.diff(leader_history) != 0)
        num_switches_list.append(num_switches)

        # GDP divergence (leader vs 2nd, mean, last)
        sorted_gdp = np.sort(final_gdp)[::-1]
        gdp_gap_leader_second.append(sorted_gdp[0] / sorted_gdp[1] if sorted_gdp[1] > 0 else np.nan)
        gdp_gap_leader_mean.append(sorted_gdp[0] / np.mean(sorted_gdp[1:]) if np.mean(sorted_gdp[1:]) > 0 else np.nan)
        gdp_gap_leader_last.append(sorted_gdp[0] / sorted_gdp[-1] if sorted_gdp[-1] > 0 else np.nan)
        # Export divergence (same logic)
        final_exports = np.array([Exports_hist[j][-1] for j in range(num_economies)])
        sorted_exports = np.sort(final_exports)[::-1]
        export_gap_leader_second.append(sorted_exports[0] / sorted_exports[1] if sorted_exports[1] > 0 else np.nan)
        export_gap_leader_mean.append(sorted_exports[0] / np.mean(sorted_exports[1:]) if np.mean(sorted_exports[1:]) > 0 else np.nan)
        export_gap_leader_last.append(sorted_exports[0] / sorted_exports[-1] if sorted_exports[-1] > 0 else np.nan)

    if verbose:
        print(f"\nFraction of runs where the t={leader_check_time} leader remains the leader at t={T}: {count_persistent/num_runs:.3f}")
        print("Distribution of which economy was the leader at t={}:".format(leader_check_time), who_leader_at_check)
        print("Distribution of which economy won at t={}:".format(T), who_won)
        print(f"Mean number of leadership transitions per run: {np.mean(num_switches_list):.2f}")
        print(f"Mean (final) GDP gap leader/second: {np.mean(gdp_gap_leader_second):.2f}")
        print(f"Mean (final) GDP gap leader/mean others: {np.mean(gdp_gap_leader_mean):.2f}")
        print(f"Mean (final) GDP gap leader/last: {np.mean(gdp_gap_leader_last):.2f}")
        print(f"Mean (final) Export gap leader/second: {np.mean(export_gap_leader_second):.2f}")
        print(f"Mean (final) Export gap leader/mean others: {np.mean(export_gap_leader_mean):.2f}")
        print(f"Mean (final) Export gap leader/last: {np.mean(export_gap_leader_last):.2f}")

    return {
        "fraction_persistent": count_persistent / num_runs,
        "who_leader_at_check": who_leader_at_check,
        "who_won": who_won,
        "mean_num_switches": np.mean(num_switches_list),
        "std_num_switches": np.std(num_switches_list),
        "gdp_gap_leader_second": gdp_gap_leader_second,
        "gdp_gap_leader_mean": gdp_gap_leader_mean,
        "gdp_gap_leader_last": gdp_gap_leader_last,
        "export_gap_leader_second": export_gap_leader_second,
        "export_gap_leader_mean": export_gap_leader_mean,
        "export_gap_leader_last": export_gap_leader_last,
        "mean_gdp_gap_leader_second": np.nanmean(gdp_gap_leader_second),
        "mean_gdp_gap_leader_mean": np.nanmean(gdp_gap_leader_mean),
        "mean_gdp_gap_leader_last": np.nanmean(gdp_gap_leader_last),
        "mean_export_gap_leader_second": np.nanmean(export_gap_leader_second),
        "mean_export_gap_leader_mean": np.nanmean(export_gap_leader_mean),
        "mean_export_gap_leader_last": np.nanmean(export_gap_leader_last)
    }

def simulate_leadership_statistics_tariff(
    num_runs=1000,
    T=100,
    num_economies=5,
    firms_per_economy=20,
    mu=0.6,
    iota=0.4,
    rho=0.6,
    alpha=0.375,
    beta=0.5,
    gamma=1.0,
    phi=1.0,
    lambda_exo=0.05,
    theta_innovator=0.0,
    theta_imitator=0.0,
    sigma=0.1,
    chi=0.5,
    kappa=0.5,
    ex_rate_shock_var=0.0001,
    delta_K=0.05,
    tariff_rate=0.10,
    tariff_start=60,
    Y0=100.0,
    w0=5.0,
    Yw0=401.0,
    Y_exo0=1.0,
    verbose=False
):
    """
    Simulate repeated runs of an agent-based model (ABM) with a dynamic tariff regime and track leadership persistence,
    GDP/export divergence, and path dependence statistics.
    ----------
    num_runs : int, optional
        Number of independent simulation runs to perform (default: 1000).
    T : int, optional
        Number of time steps per simulation run (default: 100).
    num_economies : int, optional
        Number of economies in the simulation (default: 5).
    firms_per_economy : int, optional
        Number of firms per economy (default: 20).
    mu : float, optional
        Markup parameter for firm pricing (default: 0.6).
    iota : float, optional
        Fraction of output invested by firms (default: 0.4).
    rho : float, optional
        Fraction of output allocated to R&D by firms (default: 0.6).
    alpha : float, optional
        Elasticity parameter for exports (default: 0.375).
    beta : float, optional
        Parameter for economic adjustment (default: 0.5).
    gamma : float, optional
        Wage adjustment parameter (default: 1.0).
    phi : float, optional
        Selection strength parameter (default: 1.0).
    lambda_exo : float, optional
        Exogenous growth rate of external demand (default: 0.05).
    theta_innovator : float, optional
        Innovation probability boost for innovator firms (default: 0.0).
    theta_imitator : float, optional
        Innovation probability boost for imitator firms (default: 0.0).
    sigma : float, optional
        Standard deviation of innovation shocks (default: 0.1).
    chi : float, optional
        Strength of imitation (default: 0.5).
    kappa : float, optional
        Exchange rate adjustment parameter (default: 0.5).
    ex_rate_shock_var : float, optional
        Variance of exchange rate shocks (default: 0.0001).
    delta_K : float, optional
        Capital depreciation rate (default: 0.05).
    Y0 : float, optional
        Initial GDP per economy (default: 100.0).
    w0 : float, optional
        Initial wage per economy (default: 5.0).
    Yw0 : float, optional
        Initial world output (default: 401.0).
    Y_exo0 : float, optional
        Initial exogenous demand (default: 1.0).
    leader_check_time : int, optional
        Time step at which to record the current leader (default: 60).
    verbose : bool, optional
        If True, prints summary statistics after simulation (default: False).
    Returns
    -------
    dict
        Dictionary containing:
            - "fraction_persistent": Fraction of runs where the leader at `leader_check_time` remains leader at final time.
            - "who_leader_at_check": Array counting which economy was leader at `leader_check_time`.
            - "who_won": Array counting which economy was leader at final time.
            - "mean_num_switches": Mean number of leadership transitions per run.
            - "std_num_switches": Standard deviation of leadership transitions per run.
            - "gdp_gap_leader_second": List of GDP ratios (leader/second) at final time for each run.
            - "gdp_gap_leader_mean": List of GDP ratios (leader/mean of others) at final time for each run.
            - "gdp_gap_leader_last": List of GDP ratios (leader/last) at final time for each run.
            - "export_gap_leader_second": List of export ratios (leader/second) at final time for each run.
            - "export_gap_leader_mean": List of export ratios (leader/mean of others) at final time for each run.
            - "export_gap_leader_last": List of export ratios (leader/last) at final time for each run.
            - "mean_gdp_gap_leader_second": Mean GDP gap (leader/second) across runs.
            - "mean_gdp_gap_leader_mean": Mean GDP gap (leader/mean of others) across runs.
            - "mean_gdp_gap_leader_last": Mean GDP gap (leader/last) across runs.
            - "mean_export_gap_leader_second": Mean export gap (leader/second) across runs.
            - "mean_export_gap_leader_mean": Mean export gap (leader/mean of others) across runs.
            - "mean_export_gap_leader_last": Mean export gap (leader/last) across runs.
    Notes
     - The function uses numpy for all numerical operations and random number generation.
    - Each simulation run is seeded for reproducibility.
    - Leadership is defined as the economy with the highest GDP at a given time.
    - Gaps are ratios of GDP or exports between the leader and other economies.
    -----
    """
    count_persistent = 0
    who_imposed = np.zeros(num_economies, dtype=int)
    who_won = np.zeros(num_economies, dtype=int)
    num_switches_list = []
    gdp_gap_leader_second = []
    gdp_gap_leader_mean = []
    gdp_gap_leader_last = []
    export_gap_leader_second = []
    export_gap_leader_mean = []
    export_gap_leader_last = []

    for run in range(num_runs):
        a_firm = np.ones((num_economies, firms_per_economy))
        A_firm = np.ones((num_economies, firms_per_economy))
        z_firm = np.full((num_economies, firms_per_economy), 1/(num_economies*firms_per_economy))
        K_firm = np.ones((num_economies, firms_per_economy))
        Y_firm = np.ones((num_economies, firms_per_economy))
        Pi_firm = np.ones((num_economies, firms_per_economy))
        I_firm = np.zeros((num_economies, firms_per_economy))
        R_firm = np.zeros((num_economies, firms_per_economy))
        innovators = (np.arange(firms_per_economy) % 2 == 0)
        Y_economy = np.full(num_economies, Y0)
        A_economy = np.ones(num_economies)
        E_economy = np.ones(num_economies)
        w_economy = np.full(num_economies, w0)
        Yw_prev1 = np.full(num_economies, Yw0)
        Yw_prev2 = np.full(num_economies, Yw0)
        Y_exo = Y_exo0
        ex_rate = np.ones(num_economies)
        exports = np.ones(num_economies)
        exports_prev = np.ones(num_economies)
        exports_global = exports.sum()
        exports_global_prev = exports_global

        tariff_economy = None
        np.random.seed(run)  # Reproducibility per run

        Y_hist = [[] for _ in range(num_economies)]
        Exports_hist = [[] for _ in range(num_economies)]
        leader_history = []

        for t in range(T):
            z_economy = z_firm.sum(axis=1)
            A_prev = A_economy.copy()
            A_economy = (z_firm * A_firm).sum(axis=1) / z_economy

            exports_prev[:] = exports[:]
            for j in range(num_economies):
                Yw_t = Y_economy.sum() - Y_economy[j] + Y_exo
                exports[j] = (Yw_t ** alpha) * z_economy[j]
                Exports_hist[j].append(exports[j])

            exports_global_prev = exports_global
            exports_global = exports.sum()
            global_growth = (exports_global - exports_global_prev) / exports_global_prev if exports_global_prev > 0 else 0.0

            for j in range(num_economies):
                own_growth = (exports[j] - exports_prev[j]) / exports_prev[j] if exports_prev[j] > 0 else 0.0
                rel_growth = own_growth - global_growth
                shock = np.random.normal(0, np.sqrt(ex_rate_shock_var))
                ex_rate[j] = ex_rate[j] * (1 + kappa * rel_growth) + shock

            # --- Tariff regime activates at t=tariff_start (after period 60) ---
            if t == tariff_start:
                last_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
                tariff_economy = np.argmax(last_gdp)
                who_imposed[tariff_economy] += 1

            # --- Competitiveness with dynamic Tariff (if imposed) ---
            price_firm = (1 + mu) * w_economy[:, None] / A_firm
            E_firm = np.zeros((num_economies, firms_per_economy))
            for j in range(num_economies):
                for i in range(firms_per_economy):
                    effective_price = price_firm[j, i] * ex_rate[j]
                    if tariff_economy is not None and j == tariff_economy and j != i // firms_per_economy:
                        effective_price *= (1 + tariff_rate)
                    E_firm[j, i] = 1 / effective_price
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
                    K_firm[j, i] = (1 - delta_K) * K_firm[j, i] + I_firm[j, i]

            for j in range(num_economies):
                for i in range(firms_per_economy):
                    rel_fit = E_firm[j, i] / E_global
                    z_firm[j, i] *= (1 + phi * (rel_fit - 1))

            Yw_prev2 = Yw_prev1.copy()
            Yw_prev1 = np.array([Y_economy.sum() - Y_economy[j] + Y_exo for j in range(num_economies)])
            Y_exo *= (1 + lambda_exo)

            for j in range(num_economies):
                Y_hist[j].append(Y_economy[j])
            # Track leader
            current_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
            leader_history.append(np.argmax(current_gdp))

        # --- At the end, check if the tariff-imposer is still the leader ---
        final_gdp = np.array([Y_hist[j][-1] for j in range(num_economies)])
        winner = np.argmax(final_gdp)
        who_won[winner] += 1
        if winner == tariff_economy:
            count_persistent += 1

        # Leadership switches (count transitions in leader_history)
        num_switches = np.sum(np.diff(leader_history) != 0)
        num_switches_list.append(num_switches)

        # GDP divergence (leader vs 2nd, mean, last)
        sorted_gdp = np.sort(final_gdp)[::-1]
        gdp_gap_leader_second.append(sorted_gdp[0] / sorted_gdp[1] if sorted_gdp[1] > 0 else np.nan)
        gdp_gap_leader_mean.append(sorted_gdp[0] / np.mean(sorted_gdp[1:]) if np.mean(sorted_gdp[1:]) > 0 else np.nan)
        gdp_gap_leader_last.append(sorted_gdp[0] / sorted_gdp[-1] if sorted_gdp[-1] > 0 else np.nan)
        # Export divergence (same logic)
        final_exports = np.array([Exports_hist[j][-1] for j in range(num_economies)])
        sorted_exports = np.sort(final_exports)[::-1]
        export_gap_leader_second.append(sorted_exports[0] / sorted_exports[1] if sorted_exports[1] > 0 else np.nan)
        export_gap_leader_mean.append(sorted_exports[0] / np.mean(sorted_exports[1:]) if np.mean(sorted_exports[1:]) > 0 else np.nan)
        export_gap_leader_last.append(sorted_exports[0] / sorted_exports[-1] if sorted_exports[-1] > 0 else np.nan)

    if verbose:
        print(f"\nFraction of runs where the tariff-imposer at t={tariff_start} remains the leader at t={T}: {count_persistent/num_runs:.3f}")
        print("Distribution of which economy imposed tariffs (at t={}):".format(tariff_start), who_imposed)
        print("Distribution of which economy won at t={}:".format(T), who_won)
        print(f"Mean number of leadership transitions per run: {np.mean(num_switches_list):.2f}")
        print(f"Mean (final) GDP gap leader/second: {np.mean(gdp_gap_leader_second):.2f}")
        print(f"Mean (final) GDP gap leader/mean others: {np.mean(gdp_gap_leader_mean):.2f}")
        print(f"Mean (final) GDP gap leader/last: {np.mean(gdp_gap_leader_last):.2f}")
        print(f"Mean (final) Export gap leader/second: {np.mean(export_gap_leader_second):.2f}")
        print(f"Mean (final) Export gap leader/mean others: {np.mean(export_gap_leader_mean):.2f}")
        print(f"Mean (final) Export gap leader/last: {np.mean(export_gap_leader_last):.2f}")

    return {
        "fraction_persistent": count_persistent / num_runs,
        "who_imposed": who_imposed,
        "who_won": who_won,
        "mean_num_switches": np.mean(num_switches_list),
        "std_num_switches": np.std(num_switches_list),
        "gdp_gap_leader_second": gdp_gap_leader_second,
        "gdp_gap_leader_mean": gdp_gap_leader_mean,
        "gdp_gap_leader_last": gdp_gap_leader_last,
        "export_gap_leader_second": export_gap_leader_second,
        "export_gap_leader_mean": export_gap_leader_mean,
        "export_gap_leader_last": export_gap_leader_last,
        "mean_gdp_gap_leader_second": np.nanmean(gdp_gap_leader_second),
        "mean_gdp_gap_leader_mean": np.nanmean(gdp_gap_leader_mean),
        "mean_gdp_gap_leader_last": np.nanmean(gdp_gap_leader_last),
        "mean_export_gap_leader_second": np.nanmean(export_gap_leader_second),
        "mean_export_gap_leader_mean": np.nanmean(export_gap_leader_mean),
        "mean_export_gap_leader_last": np.nanmean(export_gap_leader_last)
    }

def plot_comparative_distributions(result_tariff, result_no_tariff):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # === 1. Final Leader Identity ===
    sns.barplot(x=np.arange(len(result_tariff["who_won"])), y=result_tariff["who_won"], ax=axs[0, 0])
    axs[0, 0].set_title("Final Leaders (Tariff)")
    axs[0, 0].set_xlabel("Economy")
    axs[0, 0].set_ylabel("Count")
    
    sns.barplot(x=np.arange(len(result_no_tariff["who_won"])), y=result_no_tariff["who_won"], ax=axs[0, 1])
    axs[0, 1].set_title("Final Leaders (No Tariff)")
    axs[0, 1].set_xlabel("Economy")
    axs[0, 1].set_ylabel("Count")

    # === 2. Regime Persistence ===
    sns.barplot(
        x=["Tariff", "No Tariff"],
        y=[result_tariff["fraction_persistent"], result_no_tariff["fraction_persistent"]],
        ax=axs[0, 2]
    )
    axs[0, 2].set_title("Fraction of Persistent Leadership")
    axs[0, 2].set_ylabel("Share of Runs")
    axs[0, 2].set_ylim(0, 1)

    # === 3. GDP Gaps (Leader / 2nd Place) ===
    df_gdp = pd.DataFrame({
        "GDP Gap": result_tariff["gdp_gap_leader_second"] + result_no_tariff["gdp_gap_leader_second"],
        "Regime": ["Tariff"] * len(result_tariff["gdp_gap_leader_second"]) +
                  ["No Tariff"] * len(result_no_tariff["gdp_gap_leader_second"])
    })
    sns.boxplot(x="Regime", y="GDP Gap", data=df_gdp, ax=axs[1, 0])
    axs[1, 0].set_title("GDP Gap (Leader / 2nd Place)")

    # === 4. Export Gaps (Leader / 2nd Place) ===
    df_export = pd.DataFrame({
        "Export Gap": result_tariff["export_gap_leader_second"] + result_no_tariff["export_gap_leader_second"],
        "Regime": ["Tariff"] * len(result_tariff["export_gap_leader_second"]) +
                  ["No Tariff"] * len(result_no_tariff["export_gap_leader_second"])
    })
    sns.boxplot(x="Regime", y="Export Gap", data=df_export, ax=axs[1, 1])
    axs[1, 1].set_title("Export Gap (Leader / 2nd Place)")

    # === 5. Leadership Transitions ===
    df_trans = pd.DataFrame({
        "Switches": [result_tariff["mean_num_switches"], result_no_tariff["mean_num_switches"]],
        "Regime": ["Tariff", "No Tariff"]
    })
    sns.barplot(x="Regime", y="Switches", data=df_trans, ax=axs[1, 2])
    axs[1, 2].set_title("Mean Number of Leadership Transitions")

    plt.tight_layout()
    plt.show()
