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
Ag⁺/Cl⁻ complexation with water equilibrium at T = 320 K.

    Rxn 1:  H₂O  ⇌  H⁺ + OH⁻            log₁₀(K₂₉₈) = −13.997,  ΔHᵣ = +55.8 kJ/mol
    Rxn 2:  Ag⁺ + Cl⁻  ⇌  AgCl(aq)      log₁₀(K₂₉₈) = +3.31,   ΔHᵣ = −12 kJ/mol

H₂O is the solvent and is excluded from equilibrium expressions by convention.

The Van't Hoff correction is applied **inside the model** using the ΔHᵣ values
passed to the property package and the process temperature Var on the unit model:

    ln(K_T) = ln(K_T₀) + (ΔHᵣ/R) × (1/T₀ − 1/T)

Because there are no precipitation reactions the objective is zero and IPOPT
solves a square (feasibility) system.

This example mirrors the ``all_aqueous.ipynb`` notebook from the original
Pyomo precipitator (TorresCMULab/precipitator-unit-model, henryslaw branch).

Note on reference model
-----------------------
The original Pyomo notebook has a bug where the ``init_pressure_gas`` argument
(added later to the function signature) shifts all subsequent positional
arguments, causing ``process_temp_kelvin=320`` to land in ``MW_solvent_g_per_mol``
instead of ``temp``. As a result, the reference notebook effectively runs at
T=298.15 K without Van't Hoff correction. This IDAES example correctly applies
the Van't Hoff correction and runs at T=320 K; a corrected version of the
Pyomo notebook is provided alongside this file.
"""

import math

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator

# ============================================================================
# Thermodynamic data at T₀ = 298.15 K (reference temperature)
# Van't Hoff correction is applied inside the model using dHr_aq_dict
# ============================================================================

LN10 = math.log(10)
T = 320.0  # K (process temperature — passed to temperature Var)

# Aqueous components (mol/L); H₂O excluded — it is the solvent
AQ_COMP_LIST = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)"]

# ln(K) at T₀ = 298.15 K (reference, before Van't Hoff correction)
LN_K_AQ_DICT = {
    1: -13.997 * LN10,  # H₂O ⇌ H⁺ + OH⁻
    2:    3.31  * LN10,  # Ag⁺ + Cl⁻ ⇌ AgCl(aq)
}

# Stoichiometry: reactants are negative, products are positive
# H2O is omitted from stoich_aq_dict because it is the solvent (unit activity)
STOICH_AQ_DICT = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
}

# ΔHr values (J/mol) — used by the model for the Van't Hoff correction
DHR_AQ_DICT = {
    1: +55_800,  # H₂O ⇌ H⁺ + OH⁻        endothermic
    2: -12_000,  # Ag⁺ + Cl⁻ ⇌ AgCl(aq)  exothermic
}

# Initial aqueous concentrations (mol/L)
C0 = {
    "H+": 1e-5,
    "OH-": 1e-9,
    "Ag+": 1e-4,
    "Cl-": 1e-4,
    "AgCl(aq)": 1e-30,
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
        dHr_aq_dict=DHR_AQ_DICT,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
    )

    prec = m.fs.prec

    # Fix process temperature (single-temperature solve)
    prec.temperature.fix(T)

    # Fix aqueous inlet
    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL)
    for species, conc in C0.items():
        prec.aqueous_inlet.conc_mol_comp[0, species].fix(conc)

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

    rxn_labels = {1: "H₂O ⇌ H⁺ + OH⁻", 2: "Ag⁺ + Cl⁻ ⇌ AgCl(aq)"}

    print("\n" + "=" * 60)
    print(f"All-Aqueous Equilibrium Example  (T = {T} K)")
    print("=" * 60)
    for r, label in rxn_labels.items():
        lnk = pyo.value(prec.log_k[r])
        print(f"\n  ln(K{r}) at {T} K = {lnk:.6f}  ({label})")

    print("\nAqueous species concentrations:")
    print(f"  {'Species':<15}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in AQ_COMP_LIST:
        c_in = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:>16.4e}  {c_out:>16.4e}")

    print("\nReaction extents (mol/L):")
    rxn_labels = {1: "H₂O → H⁺ + OH⁻", 2: "Ag⁺ + Cl⁻ → AgCl(aq)"}
    for r in prec.merged_rxns:
        xe = pyo.value(prec.rxn_extent[r])
        print(f"  Rxn {r} ({rxn_labels[r]}): {xe:+.4e}")

    print("\nEquilibrium residual check (|Σα·ln(C) − ln(K)| < 1e-6):")
    for r, label in rxn_labels.items():
        lhs = sum(
            STOICH_AQ_DICT[r][i] * pyo.value(prec.log_conc_out[i])
            for i in STOICH_AQ_DICT[r]
        )
        residual = abs(lhs - pyo.value(prec.log_k[r]))
        ok = "PASS" if residual < 1e-6 else "FAIL"
        print(f"  Rxn {r} ({label}): residual = {residual:.2e}  [{ok}]")

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
