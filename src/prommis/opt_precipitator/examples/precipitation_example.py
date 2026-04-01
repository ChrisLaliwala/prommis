#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
AgCl Precipitation Example (Stage 1)
======================================

This example demonstrates the OptPrecipitator unit model with active solid
precipitation at T = 320 K and a volumetric flow rate of 2 L/s.

Chemistry
---------
    Rxn 1:  H₂O  ⇌  H⁺ + OH⁻              log₁₀(K₂₉₈) = −13.997,  ΔHᵣ = +55.8 kJ/mol
    Rxn 2:  Ag⁺ + Cl⁻  ⇌  AgCl(aq)        log₁₀(K₂₉₈) = +3.31,    ΔHᵣ = −12 kJ/mol
    Rxn 3:  AgCl(s)  ⇌  Ag⁺ + Cl⁻         log₁₀(K₂₉₈) = −9.75,    ΔHᵣ = +65.2 kJ/mol

Rxns 1–2 are aqueous equilibria (equality constraints).
Rxn 3 is a precipitation / dissolution equilibrium — the saturation inequality
drives the objective to push the system to the solubility product boundary.

The Van't Hoff correction is applied **inside the model** using the ΔHᵣ values
passed to the property package and the process temperature Var on the unit model:

    ln(K_T) = ln(K_T₀) + (ΔHᵣ/R) × (1/T₀ − 1/T)

This example mirrors Stage 1 of ``multiple_precipitator_process.ipynb`` from the
original Pyomo precipitator (TorresCMULab/precipitator-unit-model, henryslaw
branch), corrected to properly apply Van't Hoff temperature correction.
"""

import math

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter

# ============================================================================
# Thermodynamic data at T₀ = 298.15 K (reference temperature)
# Van't Hoff correction is applied inside the model using dHr dicts
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

# Stoichiometry for aqueous species in ALL reactions (including precipitation rxn 3)
# H₂O omitted (solvent, unit activity)
STOICH_AQ_DICT = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},  # dissolution gives Ag+ and Cl-
}

# ΔHr values for aqueous reactions (J/mol)
DHR_AQ_DICT = {
    1: +55_800,  # H₂O ⇌ H⁺ + OH⁻        endothermic
    2: -12_000,  # Ag⁺ + Cl⁻ ⇌ AgCl(aq)  exothermic
}

# Solid (precipitate) components
SP_COMP_LIST = ["AgCl(s)"]

# ln(Ksp) at T₀ = 298.15 K
LN_K_SP_DICT = {3: -9.75 * LN10}  # AgCl(s) ⇌ Ag⁺ + Cl⁻

# Stoichiometry for solid species in precipitation reaction 3
STOICH_SP_DICT = {3: {"AgCl(s)": -1}}  # AgCl(s) dissolves → negative

# ΔHr for precipitation reaction (J/mol)
DHR_SP_DICT = {3: +65_200}  # AgCl(s) ⇌ Ag⁺ + Cl⁻  endothermic

# -----------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------
C0 = {
    "H+": 1e-5,
    "OH-": 1e-9,
    "Ag+": 1e-4,
    "Cl-": 1e-4,
    "AgCl(aq)": 1e-30,
}

# Initial solid amount (mol/s — IDAES flow-based convention)
M0_AGCLS = 1e-5  # mol/s

# Volumetric flow rate: 2 L/s corresponds to 2 kg solvent in the batch reference
FLOW_VOL = 2.0  # L/s


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
    m.fs.sp_props = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST,
        ln_k_sp_dict=LN_K_SP_DICT,
        stoich_sp_dict=STOICH_SP_DICT,
        dHr_sp_dict=DHR_SP_DICT,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
        property_package_precipitate=m.fs.sp_props,
    )

    prec = m.fs.prec

    # Fix process temperature (single-temperature solve)
    prec.temperature.fix(T)

    # Fix aqueous inlet (molar flows = concentration × volumetric flow rate)
    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL)
    for species, conc in C0.items():
        prec.aqueous_inlet.flow_mol_comp[0, species].fix(conc * FLOW_VOL)

    # Fix precipitate inlet
    prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(M0_AGCLS)

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
    """Print a formatted results table and verify saturation index."""
    prec = m.fs.prec
    t0 = m.fs.time.first()

    print("\n" + "=" * 65)
    print(f"AgCl Precipitation Example  (T = {T} K,  flow_vol = {FLOW_VOL} L/s)")
    print("=" * 65)

    rxn_k_labels = {
        1: "H₂O ⇌ H⁺ + OH⁻",
        2: "Ag⁺ + Cl⁻ ⇌ AgCl(aq)",
        3: "AgCl(s) ⇌ Ag⁺ + Cl⁻",
    }
    print(f"\nln(K) values at T={T} K (Van't Hoff corrected by model):")
    for r, label in rxn_k_labels.items():
        lnk = pyo.value(prec.log_k[r])
        print(f"  Rxn {r}: ln(K) = {lnk:.6f}  [{label}]")

    print("\nAqueous species concentrations:")
    print(f"  {'Species':<15}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in AQ_COMP_LIST:
        c_in = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:>16.4e}  {c_out:>16.4e}")

    print("\nSolid phase (mol/s):")
    print(f"  {'Species':<15}  {'Inlet':>16}  {'Outlet':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in SP_COMP_LIST:
        m_in = pyo.value(prec.cv_precipitate.properties_in[t0].moles_precipitate_comp[sp])
        m_out = pyo.value(prec.cv_precipitate.properties_out[t0].moles_precipitate_comp[sp])
        print(f"  {sp:<15}  {m_in:>16.4e}  {m_out:>16.4e}")

    print("\nReaction extents (mol/L):")
    rxn_labels = {
        1: "H₂O → H⁺ + OH⁻",
        2: "Ag⁺ + Cl⁻ → AgCl(aq)",
        3: "AgCl(s) → Ag⁺ + Cl⁻",
    }
    for r in prec.merged_rxns:
        xe = pyo.value(prec.rxn_extent[r])
        print(f"  Rxn {r} ({rxn_labels[r]}): {xe:+.4e}")

    print("\nSaturation index check for precipitation rxn 3:")
    log_q = pyo.value(prec.log_q_sp[3])
    ln_k3 = pyo.value(prec.log_k[3])
    print(f"  ln(Q) = {log_q:.6f}")
    print(f"  ln(K) = {ln_k3:.6f}")
    print(f"  ln(K) - ln(Q) = {ln_k3 - log_q:.2e}  (should be ≈ 0 at binding)")

    print("\nAqueous equilibrium residual check (|Σα·ln(C) − ln(K)| < 1e-6):")
    for r in [1, 2]:
        lhs = sum(
            STOICH_AQ_DICT[r][i] * pyo.value(prec.log_conc_out[i])
            for i in STOICH_AQ_DICT[r]
        )
        residual = abs(lhs - pyo.value(prec.log_k[r]))
        ok = "PASS" if residual < 1e-6 else "FAIL"
        print(f"  Rxn {r}: residual = {residual:.2e}  [{ok}]")

    print("=" * 65 + "\n")


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
