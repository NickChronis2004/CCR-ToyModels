
"""
CCR Toy Models: Numerical Illustrations of Quantum Recurrence

This code accompanies the paper:
"Conditional Cosmological Recurrence in Finite Hilbert Spaces 
and Holographic Bounds within Causal Patches"
by N. Chronis and N. Sifakis, Universe (2025)

The code generates toy model simulations demonstrating:
1. Global fidelity recurrence in finite-dimensional Hilbert spaces
2. Local/coarse-grained recurrence proxies
3. Probability landscapes comparing global vs coarse-grained returns
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)


def generate_goe_hamiltonian(D, seed=None):
    """
    Generate a random Hamiltonian from the Gaussian Orthogonal Ensemble (GOE).
    
    Parameters
    ----------
    D : int
        Hilbert space dimension
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    H : ndarray
        D x D real symmetric matrix
    """
    if seed is not None:
        np.random.seed(seed)
    H = np.random.randn(D, D)
    return (H + H.T) / 2


def generate_random_spectrum(D, bandwidth=1.0, seed=None):
    """
    Generate random energy spectrum (band-limited).
    
    Parameters
    ----------
    D : int
        Number of energy levels
    bandwidth : float
        Spectral bandwidth
    seed : int, optional
        Random seed
        
    Returns
    -------
    energies : ndarray
        Sorted energy eigenvalues
    """
    if seed is not None:
        np.random.seed(seed)
    energies = np.sort(np.random.uniform(-bandwidth/2, bandwidth/2, D))
    return energies


def generate_dirichlet_weights(D, alpha=1.0, seed=None):
    """
    Generate random weights from Dirichlet distribution.
    
    Parameters
    ----------
    D : int
        Number of weights
    alpha : float
        Dirichlet concentration parameter (alpha=1 gives uniform on simplex)
    seed : int, optional
        Random seed
        
    Returns
    -------
    weights : ndarray
        Probability weights summing to 1
    """
    if seed is not None:
        np.random.seed(seed)
    weights = np.random.dirichlet(np.ones(D) * alpha)
    return weights


def compute_fidelity(energies, weights, times):
    """
    Compute fidelity F(t) = |<ψ(0)|ψ(t)>|² for a random initial state.
    
    For |ψ(0)> = Σ_j c_j |E_j>, we have:
    F(t) = |Σ_j |c_j|² exp(-i E_j t)|²
    
    Parameters
    ----------
    energies : ndarray
        Energy eigenvalues
    weights : ndarray
        |c_j|² coefficients (must sum to 1)
    times : ndarray
        Time points
        
    Returns
    -------
    fidelity : ndarray
        F(t) at each time point
    """
    # Compute phases: exp(-i E_j t) for all j and all t
    phases = np.exp(-1j * np.outer(times, energies))
    # Weighted sum
    amplitude = np.sum(weights * phases, axis=1)
    fidelity = np.abs(amplitude)**2
    return fidelity


def compute_local_proxy(frequencies, weights, times):
    """
    Compute local recurrence proxy R_A(t) for a subsystem.
    
    R_A(t) = |Σ_k q_k exp(-i ω_k t)|
    
    Parameters
    ----------
    frequencies : ndarray
        Mode frequencies
    weights : ndarray
        Mode weights
    times : ndarray
        Time points
        
    Returns
    -------
    proxy : ndarray
        R_A(t) at each time point
    """
    phases = np.exp(-1j * np.outer(times, frequencies))
    amplitude = np.sum(weights * phases, axis=1)
    proxy = np.abs(amplitude)
    return proxy


def find_recurrence_time(signal, times, threshold, mode='fidelity'):
    """
    Find first recurrence time where signal crosses threshold.
    
    Parameters
    ----------
    signal : ndarray
        Fidelity or proxy signal
    times : ndarray
        Time points
    threshold : float
        Threshold value (epsilon for fidelity, epsilon_A for proxy)
    mode : str
        'fidelity' (find F >= 1-epsilon) or 'proxy' (find R <= epsilon)
        
    Returns
    -------
    T_rec : float or None
        First recurrence time, or None if not found
    """
    if mode == 'fidelity':
        mask = signal >= (1 - threshold)
    else:  # proxy mode
        mask = signal <= threshold
    
    # Exclude t=0
    mask[0] = False
    
    if np.any(mask):
        return times[np.argmax(mask)]
    return None


def compute_recurrence_statistics(D, n_realizations=1000, t_max=100, dt=0.1, 
                                   epsilon=0.01, seed=None):
    """
    Compute recurrence time statistics over ensemble of random Hamiltonians.
    
    Parameters
    ----------
    D : int
        Hilbert space dimension
    n_realizations : int
        Number of random realizations
    t_max : float
        Maximum simulation time
    dt : float
        Time step
    epsilon : float
        Fidelity threshold
    seed : int, optional
        Base random seed
        
    Returns
    -------
    T_rec_list : list
        Recurrence times for each realization (None if not found)
    """
    times = np.arange(0, t_max, dt)
    T_rec_list = []
    
    for i in range(n_realizations):
        s = seed + i if seed is not None else None
        energies = generate_random_spectrum(D, seed=s)
        weights = generate_dirichlet_weights(D, seed=s)
        fidelity = compute_fidelity(energies, weights, times)
        T_rec = find_recurrence_time(fidelity, times, epsilon, mode='fidelity')
        T_rec_list.append(T_rec)
    
    return T_rec_list


# =============================================================================
# FIGURE 1: Global Fidelity and Local Proxy
# =============================================================================

def plot_figure1():
    """
    Generate Figure 1: Global fidelity F(t) and local recurrence proxy.
    Left: Random-spectrum model with D=128
    Right: Few-mode model with M=24
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Left panel: Global Fidelity ---
    D = 128
    times = np.arange(0, 120, 0.1)
    
    np.random.seed(42)
    energies = generate_random_spectrum(D, bandwidth=2.0, seed=42)
    weights = generate_dirichlet_weights(D, seed=42)
    fidelity = compute_fidelity(energies, weights, times)
    
    axes[0].plot(times, fidelity, 'b-', linewidth=0.5)
    axes[0].set_xlabel('t', fontsize=12)
    axes[0].set_ylabel('Fidelity F(t)', fontsize=12)
    axes[0].set_title(f'Global Fidelity (Random Finite-D Model, D={D})', fontsize=11)
    axes[0].set_xlim(0, 120)
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='ε=0.01 threshold')
    
    # --- Right panel: Local Proxy ---
    M = 24
    times_local = np.arange(0, 120, 0.1)
    
    np.random.seed(123)
    frequencies = np.random.uniform(0.5, 2.0, M)
    weights_local = generate_dirichlet_weights(M, seed=123)
    proxy = compute_local_proxy(frequencies, weights_local, times_local)
    
    # Normalize to show oscillations clearly
    proxy_normalized = proxy / np.max(proxy)
    
    axes[1].plot(times_local, proxy_normalized, 'b-', linewidth=0.5)
    axes[1].set_xlabel('t', fontsize=12)
    axes[1].set_ylabel('Local recurrence proxy', fontsize=12)
    axes[1].set_title(f'Local Recurrence Proxy (Few-Mode Model, M={M})', fontsize=11)
    axes[1].set_xlim(0, 120)
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(y=0.38, color='r', linestyle='--', alpha=0.5, label='local return ε_A=0.38')
    
    plt.tight_layout()
    plt.savefig('fig1_toy_models.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig1_toy_models.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 2: Probability of Return (Global vs Coarse-grained)
# =============================================================================

def plot_figure2():
    """
    Generate Figure 2: Cumulative recurrence probability for fixed D.
    Compares global ε-recurrence vs coarse-grained return.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    D = 64
    
    # Time axis (normalized by spectral width)
    T_normalized = np.linspace(0, 2.5e65, 1000)
    
    # Global recurrence: τ_global ~ exp(c*D), so P(T) = 1 - exp(-T/τ)
    # For D=64, use a large characteristic time
    tau_global = np.exp(0.5 * D)  # ~ exp(32) ~ 10^14
    P_global = 1 - np.exp(-T_normalized / tau_global)
    
    # Coarse-grained: τ_cg ~ exp(c*S_A) where S_A ~ log(D) << D
    S_A = np.log(D)  # ~ 4.16
    tau_cg = np.exp(0.5 * S_A)  # ~ 8
    P_coarse = 1 - np.exp(-T_normalized / (tau_cg * 1e64))
    
    ax.plot(T_normalized, P_global, 'orange', linewidth=2, 
            label=f'Global ε-recurrence (ε=0.01, D={D})')
    ax.plot(T_normalized, P_coarse, 'blue', linewidth=2,
            label=f'Coarse-grained return (S_A={S_A:.2f})')
    
    ax.set_xlabel('Time T (dimensionless units; spectral width normalized)', fontsize=11)
    ax.set_ylabel('Cumulative probability P(return by T)', fontsize=11)
    ax.set_title('Probability of return vs time: global vs coarse-grained (fixed D)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 2.5e65)
    ax.set_ylim(0, 1.0)
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('fig2_probability_fixed_D.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig2_probability_fixed_D.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# FIGURES 3 & 4: 3D Probability Landscapes
# =============================================================================

def plot_figure3_4():
    """
    Generate Figures 3 and 4: 3D probability landscapes.
    Figure 3: Global recurrence probability P(T, D)
    Figure 4: Coarse-grained recurrence probability P(T, D)
    """
    
    # --- Figure 3: Global recurrence ---
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Create mesh
    D_values = np.linspace(20, 100, 50)
    T_values = np.linspace(0, 1500, 50)
    D_mesh, T_mesh = np.meshgrid(D_values, T_values)
    
    # Global: τ(D) ~ exp(c*D), P = 1 - exp(-T/τ)
    c_global = 0.1
    tau_global = np.exp(c_global * D_mesh)
    P_global = 1 - np.exp(-T_mesh / tau_global)
    
    surf3 = ax3.plot_surface(D_mesh, T_mesh, P_global, cmap='viridis', 
                              edgecolor='none', alpha=0.9)
    ax3.set_xlabel('Hilbert-space dimension D', fontsize=10)
    ax3.set_ylabel('Time T (dimensionless)', fontsize=10)
    ax3.set_zlabel('P_global(return by T)', fontsize=10)
    ax3.set_title('Global ε-recurrence probability vs time and D\n(τ(D) ∝ exp(c D))', fontsize=11)
    ax3.view_init(elev=25, azim=-60)
    fig3.colorbar(surf3, shrink=0.5, aspect=10, label='Probability')
    
    plt.tight_layout()
    plt.savefig('fig3_global_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig3_global_landscape.pdf', bbox_inches='tight')
    
    # --- Figure 4: Coarse-grained recurrence ---
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    # Coarse-grained mesh (different time scale)
    T_cg_values = np.linspace(0, 3.0, 50)
    D_mesh_cg, T_mesh_cg = np.meshgrid(D_values, T_cg_values)
    
    # Coarse-grained: τ(D) ~ exp(c*S_A), S_A = f*log(D)
    f = 1.0  # fraction of log D for subsystem entropy
    c_cg = 0.5
    S_A = f * np.log(D_mesh_cg)
    tau_cg = np.exp(c_cg * S_A)
    P_coarse = 1 - np.exp(-T_mesh_cg / tau_cg)
    
    surf4 = ax4.plot_surface(D_mesh_cg, T_mesh_cg, P_coarse, cmap='viridis',
                              edgecolor='none', alpha=0.9)
    ax4.set_xlabel('Hilbert-space dimension D', fontsize=10)
    ax4.set_ylabel('Time T (dimensionless)', fontsize=10)
    ax4.set_zlabel('P_coarse(return by T)', fontsize=10)
    ax4.set_title('Coarse-grained recurrence probability vs time and D\n(τ_cg(D) ∝ exp(c S_A), S_A = f log D)', fontsize=11)
    ax4.view_init(elev=25, azim=-60)
    fig4.colorbar(surf4, shrink=0.5, aspect=10, label='Probability')
    
    plt.tight_layout()
    plt.savefig('fig4_coarse_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig4_coarse_landscape.pdf', bbox_inches='tight')
    
    plt.show()
    
    return fig3, fig4


# =============================================================================
# TABLE 2: Recurrence Statistics
# =============================================================================

def compute_table2_statistics():
    """
    Compute statistics for Table 2 in the paper.
    
    Returns median recurrence times for:
    - T_rec^(10^-2): global with ε=0.01
    - T_rec^(10^-3): global with ε=0.001
    - T_rec^(A): local proxy with ε_A=0.1
    - t_scr: scrambling time (90% of plateau)
    """
    print("Computing Table 2 statistics...")
    print("=" * 60)
    
    D = 128
    M = 24
    n_realizations = 1000
    t_max = 100
    dt = 0.1
    times = np.arange(0, t_max, dt)
    
    T_rec_001 = []
    T_rec_0001 = []
    T_rec_local = []
    t_scrambling = []
    
    for i in range(n_realizations):
        # Global fidelity
        np.random.seed(i)
        energies = generate_random_spectrum(D, bandwidth=2.0, seed=i)
        weights = generate_dirichlet_weights(D, seed=i)
        fidelity = compute_fidelity(energies, weights, times)
        
        T1 = find_recurrence_time(fidelity, times, 0.01, mode='fidelity')
        T2 = find_recurrence_time(fidelity, times, 0.001, mode='fidelity')
        
        T_rec_001.append(T1)
        T_rec_0001.append(T2)
        
        # Local proxy
        np.random.seed(i + 10000)
        frequencies = np.random.uniform(0.5, 2.0, M)
        weights_local = generate_dirichlet_weights(M, seed=i + 10000)
        proxy = compute_local_proxy(frequencies, weights_local, times)
        proxy_norm = proxy / proxy[0]  # Normalize by initial value
        
        T_local = find_recurrence_time(proxy_norm, times, 0.1, mode='proxy')
        T_rec_local.append(T_local)
        
        # Scrambling time: when proxy drops to 10% of initial (i.e., 90% scrambled)
        scramble_mask = proxy_norm <= 0.1
        scramble_mask[0] = False
        if np.any(scramble_mask):
            t_scrambling.append(times[np.argmax(scramble_mask)])
        else:
            t_scrambling.append(None)
    
    # Compute medians (excluding None values)
    T_rec_001_valid = [t for t in T_rec_001 if t is not None]
    T_rec_0001_valid = [t for t in T_rec_0001 if t is not None]
    T_rec_local_valid = [t for t in T_rec_local if t is not None]
    t_scrambling_valid = [t for t in t_scrambling if t is not None]
    
    print(f"D = {D}, M = {M}, realizations = {n_realizations}")
    print("-" * 60)
    print(f"T_rec^(ε=0.01):  median = {np.median(T_rec_001_valid):.1f} "
          f"(found in {len(T_rec_001_valid)}/{n_realizations} cases)")
    print(f"T_rec^(ε=0.001): median = {np.median(T_rec_0001_valid) if T_rec_0001_valid else 'n/a'} "
          f"(found in {len(T_rec_0001_valid)}/{n_realizations} cases)")
    print(f"T_rec^(A) (ε_A=0.1): median = {np.median(T_rec_local_valid):.1f} "
          f"(found in {len(T_rec_local_valid)}/{n_realizations} cases)")
    print(f"t_scr (90% plateau): median = {np.median(t_scrambling_valid):.1f} "
          f"(found in {len(t_scrambling_valid)}/{n_realizations} cases)")
    print("=" * 60)
    
    return {
        'T_rec_001': np.median(T_rec_001_valid) if T_rec_001_valid else None,
        'T_rec_0001': np.median(T_rec_0001_valid) if T_rec_0001_valid else None,
        'T_rec_local': np.median(T_rec_local_valid) if T_rec_local_valid else None,
        't_scr': np.median(t_scrambling_valid) if t_scrambling_valid else None
    }


# =============================================================================
# LEVEL STATISTICS (Wigner-Dyson verification)
# =============================================================================

def verify_level_statistics(D=128, n_samples=100):
    """
    Verify that GOE Hamiltonians exhibit Wigner-Dyson level spacing.
    
    The nearest-neighbor spacing distribution for GOE should follow:
    P(s) = (π/2) s exp(-π s²/4)
    
    Parameters
    ----------
    D : int
        Hilbert space dimension
    n_samples : int
        Number of Hamiltonian samples
    """
    all_spacings = []
    
    for i in range(n_samples):
        H = generate_goe_hamiltonian(D, seed=i)
        energies = np.linalg.eigvalsh(H)
        spacings = np.diff(energies)
        # Normalize by mean spacing (unfolding)
        spacings_normalized = spacings / np.mean(spacings)
        all_spacings.extend(spacings_normalized)
    
    all_spacings = np.array(all_spacings)
    
    # Plot histogram vs Wigner-Dyson
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.hist(all_spacings, bins=50, density=True, alpha=0.7, 
            label='GOE numerical', color='steelblue')
    
    s = np.linspace(0, 4, 200)
    wigner_dyson = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    ax.plot(s, wigner_dyson, 'r-', linewidth=2, label='Wigner-Dyson: P(s) = (π/2)s exp(-πs²/4)')
    
    ax.set_xlabel('Normalized spacing s', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'Level Spacing Distribution (D={D}, {n_samples} samples)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    
    plt.tight_layout()
    plt.savefig('level_statistics_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("CCR Toy Models - Generating Figures")
    print("=" * 60)
    
    # Generate all figures
    print("\n[1/4] Generating Figure 1: Fidelity and Local Proxy...")
    plot_figure1()
    
    print("\n[2/4] Generating Figure 2: Probability (fixed D)...")
    plot_figure2()
    
    print("\n[3/4] Generating Figures 3 & 4: 3D Landscapes...")
    plot_figure3_4()
    
    print("\n[4/4] Computing Table 2 Statistics...")
    stats = compute_table2_statistics()
    
    print("\n[Bonus] Verifying level statistics (Wigner-Dyson)...")
    verify_level_statistics()
    
    print("\nAll figures saved!")
    print("Files: fig1_toy_models.png/pdf, fig2_probability_fixed_D.png/pdf,")
    print("       fig3_global_landscape.png/pdf, fig4_coarse_landscape.png/pdf,")
    print("       level_statistics_verification.png")