#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
CO₂ / Carbonic Acid Gas-Liquid Equilibrium Example
====================================================

This example demonstrates the OptPrecipitator unit model with a gas phase,
modelling the CO₂/carbonic acid system at T = 298.15 K.

Chemistry
---------
    Rxn 1:  H₂O          ⇌  H⁺ + OH⁻             log₁₀(K₂₉₈) = −13.997
    Rxn 2:  CO₂(aq)+H₂O  ⇌  H⁺ + HCO₃⁻           log₁₀(K₂₉₈) = −6.3456
    Rxn 3:  HCO₃⁻        ⇌  H⁺ + CO₃²⁻            log₁₀(K₂₉₈) = −10.3354
    Rxn 4:  CO₂(aq)      ⇌  CO₂(g)                 log₁₀(K₂₉₈) = +3.245266 (Henry's Law)

Rxns 1–3 are aqueous equilibria; Rxn 4 is the gas-liquid (Henry's Law) equilibrium.
H₂O is the solvent and is excluded from all equilibrium expressions.
No precipitation reactions — the objective is zero (feasibility problem).

Reaction set derivation from Carbonic_minteq.csv
-------------------------------------------------
The Pyomo reference notebook (test_notebook.ipynb) uses the reactions as
written in the CSV:
    Rxn A: 2H⁺ + CO₃²⁻ ⇌ H₂O + CO₂(aq),  log₁₀(K) = +16.681
    Rxn B: H⁺  + CO₃²⁻ ⇌ HCO₃⁻,           log₁₀(K) = +10.3354

The reactions used here are algebraically equivalent combinations:
    Rxn 2 = −Rxn A + Rxn B:  CO₂(aq)+H₂O ⇌ H⁺+HCO₃⁻,  log₁₀(K) = −16.681+10.3354 = −6.3456
    Rxn 3 = −Rxn B:           HCO₃⁻ ⇌ H⁺+CO₃²⁻,         log₁₀(K) = −10.3354

These combinations eliminate the large-magnitude cancellation in the CO₃²⁻ mass
balance that arises from the CSV reaction set (where rxn_extent ≈ 7.5×10⁻⁵ for
rxns A and B and the difference is only ~3×10⁻¹² for CO₃²⁻), allowing IPOPT to
resolve the trace species (CO₃²⁻ ~ 4.6×10⁻¹¹, OH⁻ ~ 7.2×10⁻¹¹) accurately.

Since T = 298.15 K (reference temperature), no Van't Hoff correction is needed:
    ln(K) = log₁₀(K) × ln(10)

Gas property package parameters:
    rho_solvent = 1000 g/L,  MW_solvent = 18 g/mol  →  ρ/MW = 55.56 mol/L

This example mirrors ``test_notebook.ipynb`` from the original Pyomo precipitator
(TorresCMULab/precipitator-unit-model, henryslaw branch).

Expected result: CO₂(g) partial pressure ≈ 1.384 bar
"""

import math

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.gas_properties import GasParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter

# ============================================================================
# Thermodynamic data (T = 298.15 K — no Van't Hoff correction needed)
# ============================================================================

LN10 = math.log(10)
T = 298.15  # K

# Rxn 1: H₂O ⇌ H⁺ + OH⁻                         log₁₀(K) = -13.997
LN_K_RXN1 = -13.997 * LN10
# Rxn 2: CO₂(aq) + H₂O ⇌ H⁺ + HCO₃⁻             log₁₀(K) = -6.3456  (= -16.681 + 10.3354)
LN_K_RXN2 = (-16.681 + 10.3354) * LN10
# Rxn 3: HCO₃⁻ ⇌ H⁺ + CO₃²⁻                      log₁₀(K) = -10.3354  (= -Rxn B from CSV)
LN_K_RXN3 = -10.3354 * LN10
# Rxn 4: CO₂(aq) ⇌ CO₂(g)                          log₁₀(K) = 3.245266 (Henry's law)
LN_K_RXN4 = 3.245265839 * LN10

# ============================================================================
# Species and reaction data
# ============================================================================

AQ_COMP_LIST = ["CO3(2-)", "H+", "CO2(aq)", "HCO3-", "OH-"]

LN_K_AQ_DICT = {
    1: LN_K_RXN1,
    2: LN_K_RXN2,
    3: LN_K_RXN3,
}

# Stoichiometry for aqueous species in all reactions (H₂O excluded as solvent)
STOICH_AQ_DICT = {
    1: {"H+": 1, "OH-": 1},
    2: {"CO2(aq)": -1, "H+": 1, "HCO3-": 1},
    3: {"HCO3-": -1, "H+": 1, "CO3(2-)": 1},
    4: {"CO2(aq)": -1},  # aqueous species in gas rxn 4
}

# Gas phase
GAS_COMP_LIST = ["CO2(g)"]
LN_K_GAS_DICT = {4: LN_K_RXN4}
STOICH_GAS_DICT = {4: {"CO2(aq)": -1, "CO2(g)": +1}}

# Solvent properties for Henry's law: ρ/MW = 1000/18 = 55.56 mol/L
RHO_SOLVENT = 1000.0  # g/L
MW_SOLVENT = 18.0     # g/mol

# Precipitate package — required by OptPrecipitator; no solids in this example
SP_COMP_LIST = ["dummy(s)"]
LN_K_SP_DICT = {}
STOICH_SP_DICT = {}

# ============================================================================
# Initial conditions (MINTEQ values — near equilibrium for fast convergence)
# ============================================================================

C0 = {
    "CO3(2-)": 4.9439e-11,
    "H+":      2.1582e-4,
    "CO2(aq)": 9.9484e-2,
    "HCO3-":   2.1582e-4,
    "OH-":     4.8019e-11,
}

# Gas inlet: trace amount — equilibrium pressure is found by Henry's law
NG0_CO2G = 1e-20  # mol/s (at lower bound; equilibrium drives outlet to ~0.056 mol/s)

FLOW_VOL = 1.0  # L/s


# ============================================================================
# Build model
# ============================================================================


def build_model():
    """Construct, initialise, and return the IDAES model."""
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
    m.fs.gas_props = GasParameter(
        gas_comp_list=GAS_COMP_LIST,
        ln_k_gas_dict=LN_K_GAS_DICT,
        stoich_gas_dict=STOICH_GAS_DICT,
        rho_solvent=RHO_SOLVENT,
        MW_solvent=MW_SOLVENT,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
        property_package_precipitate=m.fs.sp_props,
        property_package_gas=m.fs.gas_props,
        temperature=T,
    )

    prec = m.fs.prec

    # Fix aqueous inlet
    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL)
    for species, conc in C0.items():
        prec.aqueous_inlet.conc_mol_comp[0, species].fix(conc)

    # Fix precipitate inlet (dummy solid at trace amount)
    prec.precipitate_inlet.moles_precipitate_comp[0, "dummy(s)"].fix(1e-20)

    # Fix gas inlet
    prec.gas_inlet.moles_gas_comp[0, "CO2(g)"].fix(NG0_CO2G)

    # Initialise log_conc_out from inlet values to avoid near-zero variable issues
    for sp in AQ_COMP_LIST:
        prec.log_conc_out[sp].set_value(math.log(max(C0[sp], 1e-20)))

    # Initialise outlet aqueous state from inlet
    t0 = m.fs.time.first()
    for sp in AQ_COMP_LIST:
        prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp].set_value(
            max(C0[sp], 1e-20)
        )

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
    """Print formatted results and verify against Pyomo reference."""
    prec = m.fs.prec
    t0 = m.fs.time.first()

    print("\n" + "=" * 65)
    print(f"CO₂/Carbonic Acid Gas-Liquid Example  (T = {T} K)")
    print("=" * 65)

    print("\nln(K) values (T=298.15 K, no Van't Hoff correction):")
    for r, (lnk, label) in {
        1: (LN_K_RXN1, "H₂O ⇌ H⁺+OH⁻"),
        2: (LN_K_RXN2, "CO₂(aq)+H₂O ⇌ H⁺+HCO₃⁻"),
        3: (LN_K_RXN3, "HCO₃⁻ ⇌ H⁺+CO₃²⁻"),
        4: (LN_K_RXN4, "CO₂(aq) ⇌ CO₂(g) [Henry]"),
    }.items():
        print(f"  Rxn {r}: ln(K) = {lnk:+.6f}  [{label}]")

    pyomo_ref = {
        "CO3(2-)": 4.619553e-11,
        "H+":      1.404529e-4,
        "CO2(aq)": 4.371813e-2,
        "HCO3-":   1.404529e-4,
        "OH-":     7.169175e-11,
    }

    print("\nAqueous species concentrations:")
    print(f"  {'Species':<12}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}  {'Pyomo ref':>14}  {'Rel diff':>10}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*16}  {'-'*14}  {'-'*10}")
    for sp in AQ_COMP_LIST:
        c_in  = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        ref   = pyomo_ref[sp]
        rel   = abs(c_out - ref) / ref
        print(f"  {sp:<12}  {c_in:>16.4e}  {c_out:>16.4e}  {ref:>14.4e}  {rel:>10.2e}")

    p_co2 = pyo.value(prec.partial_pressure["CO2(g)"])
    ref_p = 1.384203
    rel_p = abs(p_co2 - ref_p) / ref_p
    ok_p  = "PASS" if rel_p < 1e-3 else "FAIL"
    print(f"\nCO₂(g) partial pressure:  {p_co2:.6f} bar")
    print(f"Pyomo reference:          {ref_p:.6f} bar")
    print(f"Relative difference:      {rel_p:.2e}  [{ok_p}]")

    print("\nAqueous equilibrium residual check (|Σα·ln(C) − ln(K)| < 1e-6):")
    for r in [1, 2, 3]:
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
