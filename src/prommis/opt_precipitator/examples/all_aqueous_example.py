#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
All-Aqueous Equilibrium Example
================================

This example demonstrates the OptPrecipitator unit model for a purely aqueous
system with no solid precipitation and no gas phase.

Chemistry
---------
Ag⁺/Cl⁻ complexation in the presence of H⁺ at T = 320 K.

    Rxn 1:  Ag⁺ + Cl⁻  ⇌  AgCl(aq)      log₁₀(K₂₉₈) = 3.31,  ΔHᵣ = −12 kJ/mol
    Rxn 2:  H⁺  + Cl⁻  ⇌  HCl(aq)       log₁₀(K₂₉₈) = 0.70,  ΔHᵣ = 0 kJ/mol

Van't Hoff correction applied by the caller:

    ln(K_T) = ln(K_T₀) + (ΔHᵣ/R) × (1/T₀ − 1/T)

Because there are no precipitation reactions the objective is zero and IPOPT
solves a square (feasibility) system.

This example mirrors the ``all_aqueous.ipynb`` notebook from the original
Pyomo precipitator (TorresCMULab/precipitator-unit-model, henryslaw branch).
"""

import math

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter

# ============================================================================
# Thermodynamic data (pre-corrected to T = 320 K)
# ============================================================================

LN10 = math.log(10)
R_GAS = 8.314  # J/mol/K
T0 = 298.15  # K (reference temperature)
T = 320.0  # K (process temperature)


def vant_hoff_correction(log10_k_ref, dHr_J_per_mol):
    """Return ln(K) at T using the Van't Hoff equation."""
    ln_k_ref = log10_k_ref * LN10
    correction = (dHr_J_per_mol / R_GAS) * (1.0 / T0 - 1.0 / T)
    return ln_k_ref + correction


#  Rxn 1: Ag⁺ + Cl⁻ ⇌ AgCl(aq)   log₁₀(K₂₉₈) = 3.31,  ΔHᵣ = −12 kJ/mol
LN_K_RXN1 = vant_hoff_correction(3.31, -12_000)
#  Rxn 2: H⁺  + Cl⁻ ⇌ HCl(aq)    log₁₀(K₂₉₈) = 0.70,  ΔHᵣ = 0 kJ/mol
LN_K_RXN2 = vant_hoff_correction(0.70, 0.0)

# Aqueous components (mol/L)
AQ_COMP_LIST = ["Ag+", "Cl-", "AgCl(aq)", "H+", "OH-", "HCl(aq)"]

# ln(K) for each reaction (already corrected to 320 K)
LN_K_AQ_DICT = {
    1: LN_K_RXN1,  # Ag+ + Cl- -> AgCl(aq)
    2: LN_K_RXN2,  # H+  + Cl- -> HCl(aq)
}

# Stoichiometry: reactants are negative, products are positive
# stoich_aq_dict covers all reactions in which aqueous species participate
STOICH_AQ_DICT = {
    1: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    2: {"H+": -1, "Cl-": -1, "HCl(aq)": 1},
}

# Precipitate package — required by OptPrecipitator; no solid reactions here
SP_COMP_LIST = ["AgCl(s)"]  # dummy solid, never forms in this scenario
LN_K_SP_DICT = {}  # no precipitation reactions
STOICH_SP_DICT = {}

# Initial aqueous concentrations (mol/L)
C0 = {
    "Ag+": 1e-4,
    "Cl-": 1e-4,
    "AgCl(aq)": 1e-20,
    "H+": 1e-5,
    "OH-": 1e-9,
    "HCl(aq)": 1e-20,
}

FLOW_VOL = 1.0  # L/s


# ============================================================================
# Build model
# ============================================================================


def build_model():
    """Construct and return the IDAES model."""
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.aq_props = AqueousParameter(
        aqueous_comp_list=AQ_COMP_LIST,
        ln_k_aq_dict=LN_K_AQ_DICT,
        stoich_aq_dict=STOICH_AQ_DICT,
    )
    m.fs.sp_props = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST,
        ln_k_sp_dict=LN_K_SP_DICT,
        stoich_sp_dict=STOICH_SP_DICT,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
        property_package_precipitate=m.fs.sp_props,
        temperature=T,
    )

    prec = m.fs.prec

    # Fix aqueous inlet
    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL)
    for species, conc in C0.items():
        prec.aqueous_inlet.conc_mol_comp[0, species].fix(conc)

    # Fix precipitate inlet (dummy solid at trace amounts)
    prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(1e-20)

    return m


# ============================================================================
# Solve
# ============================================================================


def solve_model(m):
    """Solve with IPOPT and return the solver result."""
    solver = pyo.SolverFactory("ipopt")
    solver.options["nlp_scaling_method"] = "user-scaling"
    solver.options["tol"] = 1e-8
    solver.options["max_iter"] = 10_000
    result = solver.solve(m, tee=False)
    return result


# ============================================================================
# Results
# ============================================================================


def print_results(m):
    """Print a formatted results table."""
    prec = m.fs.prec
    t0 = m.fs.time.first()

    print("\n" + "=" * 60)
    print(f"All-Aqueous Equilibrium Example  (T = {T} K)")
    print("=" * 60)

    print("\nAqueous species concentrations:")
    print(f"  {'Species':<15}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in AQ_COMP_LIST:
        c_in = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:>16.4e}  {c_out:>16.4e}")

    print("\nReaction extents (mol/L):")
    for r in prec.merged_rxns:
        xe = pyo.value(prec.rxn_extent[r])
        print(f"  Rxn {r}: {xe:+.4e}")

    print("\nEquilibrium residual check (|Σα·ln(C) − ln(K)| < 1e-6):")
    for r, rxn_label in zip([1, 2], ["Ag+ + Cl- = AgCl(aq)", "H+ + Cl- = HCl(aq)"]):
        lhs = sum(
            STOICH_AQ_DICT[r][i] * pyo.value(prec.log_conc_out[i])
            for i in STOICH_AQ_DICT[r]
        )
        residual = abs(lhs - pyo.value(prec.log_k[r]))
        ok = "PASS" if residual < 1e-6 else "FAIL"
        print(f"  Rxn {r} ({rxn_label}): residual = {residual:.2e}  [{ok}]")

    print("=" * 60 + "\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    m = build_model()
    result = solve_model(m)

    status = result.solver.termination_condition
    print(f"\nSolver status: {status}")

    if str(status) == "optimal":
        print_results(m)
    else:
        print("WARNING: solver did not converge — check model formulation.")
