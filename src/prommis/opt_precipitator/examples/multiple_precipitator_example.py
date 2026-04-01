#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Two-Stage Precipitator Flowsheet Example
==========================================

This example demonstrates a two-stage flowsheet using the OptPrecipitator unit model,
replicating the results from the Pyomo reference notebook
``multiple_precipitator_process.ipynb`` (TorresCMULab/precipitator-unit-model,
henryslaw branch).

Flowsheet overview
------------------
Stage 1 — AgCl precipitation at T = 320 K, flow_vol = 2.0 L/s
    Ag⁺/Cl⁻ in near-neutral water. AgCl precipitates out.

Stage 2 — Ag₂C₂O₄ precipitation at T = 300 K, flow_vol = 3.0 L/s
    Stage 1 aqueous outlet (concentrations carried over directly) plus
    oxalic acid (H₂C₂O₄, 1×10⁻⁵ mol/L added). Ag₂C₂O₄ precipitates.

Chemistry
---------
Stage 1 reactions:
    Rxn 1: H₂O ⇌ H⁺ + OH⁻                     log₁₀(K₂₉₈) = −13.997,  ΔHᵣ = +55,800 J/mol
    Rxn 2: Ag⁺ + Cl⁻ ⇌ AgCl(aq)               log₁₀(K₂₉₈) = +3.31,    ΔHᵣ = −12,000 J/mol
    Rxn 3: AgCl(s) ⇌ Ag⁺ + Cl⁻                log₁₀(K₂₉₈) = −9.75,    ΔHᵣ = +65,200 J/mol

Stage 2 additional reactions (rxns 1–3 as above, plus):
    Rxn 4: H₂C₂O₄(aq) ⇌ 2H⁺ + C₂O₄²⁻         log₁₀(K₂₉₈) = −1.119,   ΔHᵣ = +50,000 J/mol
    Rxn 5: Ag₂C₂O₄(s) ⇌ 2Ag⁺ + C₂O₄²⁻        log₁₀(K₂₉₈) = −3.41,    ΔHᵣ = +25,000 J/mol

Sequential solve connection strategy
--------------------------------------
The two stages are modelled as independent optimisation problems. After Stage 1
converges, its outlet concentrations are read with ``pyo.value()`` and fixed
directly as the Stage 2 aqueous inlet boundary conditions. There are no IDAES
Arcs connecting the stages.

This matches the Pyomo reference notebook exactly. A fully-connected IDAES
flowsheet with Arcs would require a custom translator block because
``AqueousStateBlock`` uses concentration-based state variables
(``flow_vol`` + ``conc_mol_comp``) rather than the molar-flow-based variables
expected by the standard IDAES ``Mixer``.

``flow_vol`` is independently fixed per stage (2.0 L/s in Stage 1, 3.0 L/s in
Stage 2), matching the ``total_solvent_kg`` values (2 and 3 kg) in the Pyomo
notebook for water density 1000 g/L.
"""

import math

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter

# ============================================================================
# Thermodynamic data (T₀ = 298.15 K reference; Van't Hoff applied inside model)
# ============================================================================

LN10 = math.log(10)
T_REF = 298.15  # K

# --------------------------------------------------------------------------
# Stage 1 — T = 320 K, flow_vol = 2.0 L/s
# --------------------------------------------------------------------------

T_S1 = 320.0       # K
FLOW_VOL_S1 = 2.0  # L/s

AQ_COMP_LIST_S1 = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)"]

LN_K_AQ_DICT_S1 = {
    1: -13.997 * LN10,  # H₂O ⇌ H⁺ + OH⁻
    2:    3.31  * LN10,  # Ag⁺ + Cl⁻ ⇌ AgCl(aq)
}

STOICH_AQ_DICT_S1 = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},   # aqueous part of precipitation rxn 3
}

DHR_AQ_DICT_S1 = {
    1: +55_800,  # J/mol
    2: -12_000,  # J/mol
}

SP_COMP_LIST_S1 = ["AgCl(s)"]

LN_K_SP_DICT_S1 = {3: -9.75 * LN10}

STOICH_SP_DICT_S1 = {3: {"AgCl(s)": -1}}

DHR_SP_DICT_S1 = {3: +65_200}  # J/mol

C0_S1 = {
    "H+":       1e-5,
    "OH-":      1e-9,
    "Ag+":      1e-4,
    "Cl-":      1e-4,
    "AgCl(aq)": 1e-30,
}

M0_AGCLS_S1 = 1e-5  # mol/s (initial solid AgCl)

# --------------------------------------------------------------------------
# Stage 2 — T = 300 K, flow_vol = 3.0 L/s
# --------------------------------------------------------------------------

T_S2 = 300.0       # K
FLOW_VOL_S2 = 3.0  # L/s

# Stage 2 includes all Stage 1 aqueous species plus two new ones
AQ_COMP_LIST_S2 = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)", "H2C2O4(aq)", "C2O4^{2-}"]

LN_K_AQ_DICT_S2 = {
    1: -13.997 * LN10,  # H₂O ⇌ H⁺ + OH⁻
    2:    3.31  * LN10,  # Ag⁺ + Cl⁻ ⇌ AgCl(aq)
    4:   -1.119 * LN10,  # H₂C₂O₄(aq) ⇌ 2H⁺ + C₂O₄²⁻
}

STOICH_AQ_DICT_S2 = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},                            # aqueous part of rxn 3
    4: {"H2C2O4(aq)": -1, "H+": 2, "C2O4^{2-}": 1},
    5: {"Ag+": 2, "C2O4^{2-}": 1},                     # aqueous part of rxn 5
}

DHR_AQ_DICT_S2 = {
    1: +55_800,  # J/mol
    2: -12_000,  # J/mol
    4: +50_000,  # J/mol
}

SP_COMP_LIST_S2 = ["AgCl(s)", "Ag2C2O4(s)"]

LN_K_SP_DICT_S2 = {
    3: -9.75 * LN10,   # AgCl(s) ⇌ Ag⁺ + Cl⁻
    5: -3.41 * LN10,   # Ag₂C₂O₄(s) ⇌ 2Ag⁺ + C₂O₄²⁻
}

STOICH_SP_DICT_S2 = {
    3: {"AgCl(s)": -1},
    5: {"Ag2C2O4(s)": -1},
}

DHR_SP_DICT_S2 = {
    3: +65_200,  # J/mol
    5: +25_000,  # J/mol
}

# New species added in Stage 2 (not carried from Stage 1)
C0_S2_NEW = {
    "H2C2O4(aq)": 1e-5,
    "C2O4^{2-}":  1e-30,
}

M0_AG2C2O4_S2 = 1e-30  # mol/s (trace initial Ag₂C₂O₄)


# ============================================================================
# Stage 1: build
# ============================================================================


def build_stage1_model():
    """Construct and return the solved Stage 1 IDAES model."""
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.aq_props = AqueousParameter(
        aqueous_comp_list=AQ_COMP_LIST_S1,
        ln_k_aq_dict=LN_K_AQ_DICT_S1,
        stoich_aq_dict=STOICH_AQ_DICT_S1,
        dHr_aq_dict=DHR_AQ_DICT_S1,
    )
    m.fs.sp_props = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST_S1,
        ln_k_sp_dict=LN_K_SP_DICT_S1,
        stoich_sp_dict=STOICH_SP_DICT_S1,
        dHr_sp_dict=DHR_SP_DICT_S1,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
        property_package_precipitate=m.fs.sp_props,
    )

    prec = m.fs.prec
    prec.temperature.fix(T_S1)

    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL_S1)
    for sp, conc in C0_S1.items():
        prec.aqueous_inlet.conc_mol_comp[0, sp].fix(conc)

    prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(M0_AGCLS_S1)

    # Initialise log_conc_out from inlet values
    for sp in AQ_COMP_LIST_S1:
        prec.log_conc_out[sp].set_value(math.log(max(C0_S1[sp], 1e-20)))

    # Initialise outlet aqueous state from inlet
    t0 = m.fs.time.first()
    for sp in AQ_COMP_LIST_S1:
        prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp].set_value(
            max(C0_S1[sp], 1e-20)
        )

    return m


# ============================================================================
# Stage 2: build
# ============================================================================


def build_stage2_model(m1):
    """Construct and return the Stage 2 IDAES model.

    Reads Stage 1 outlet concentrations from *m1* (must be solved) and fixes
    them as Stage 2 aqueous inlet boundary conditions. The ``flow_vol`` is
    independently fixed at 3.0 L/s.
    """
    t0_s1 = m1.fs.time.first()
    prec1 = m1.fs.prec

    # Extract Stage 1 outlet aqueous concentrations
    c_out_s1 = {
        sp: pyo.value(prec1.cv_aqueous.properties_out[t0_s1].conc_mol_comp[sp])
        for sp in AQ_COMP_LIST_S1
    }

    # Extract Stage 1 outlet solid amount
    m_agcls_s1 = pyo.value(
        prec1.cv_precipitate.properties_out[t0_s1].moles_precipitate_comp["AgCl(s)"]
    )

    # Build Stage 2 inlet concentrations (Stage 1 species + new species)
    c0_s2 = {sp: c_out_s1[sp] for sp in AQ_COMP_LIST_S1}
    c0_s2.update(C0_S2_NEW)

    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.aq_props = AqueousParameter(
        aqueous_comp_list=AQ_COMP_LIST_S2,
        ln_k_aq_dict=LN_K_AQ_DICT_S2,
        stoich_aq_dict=STOICH_AQ_DICT_S2,
        dHr_aq_dict=DHR_AQ_DICT_S2,
    )
    m.fs.sp_props = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST_S2,
        ln_k_sp_dict=LN_K_SP_DICT_S2,
        stoich_sp_dict=STOICH_SP_DICT_S2,
        dHr_sp_dict=DHR_SP_DICT_S2,
    )

    m.fs.prec = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props,
        property_package_precipitate=m.fs.sp_props,
    )

    prec = m.fs.prec
    prec.temperature.fix(T_S2)

    prec.aqueous_inlet.flow_vol[0].fix(FLOW_VOL_S2)
    for sp in AQ_COMP_LIST_S2:
        prec.aqueous_inlet.conc_mol_comp[0, sp].fix(max(c0_s2[sp], 1e-30))

    prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(
        max(m_agcls_s1, 1e-30)
    )
    prec.precipitate_inlet.moles_precipitate_comp[0, "Ag2C2O4(s)"].fix(M0_AG2C2O4_S2)

    # Initialise log_conc_out from inlet values
    for sp in AQ_COMP_LIST_S2:
        prec.log_conc_out[sp].set_value(math.log(max(c0_s2[sp], 1e-20)))

    # Initialise outlet aqueous state from inlet
    t0 = m.fs.time.first()
    for sp in AQ_COMP_LIST_S2:
        prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp].set_value(
            max(c0_s2[sp], 1e-20)
        )

    return m


# ============================================================================
# Solve
# ============================================================================


def solve_model(m, tol=1e-8):
    """Solve with IPOPT and return the solver result."""
    solver = pyo.SolverFactory("ipopt")
    solver.options["nlp_scaling_method"] = "user-scaling"
    solver.options["tol"] = tol
    solver.options["max_iter"] = 10_000
    result = solver.solve(m, tee=False)
    return result


# ============================================================================
# Results
# ============================================================================


def print_results(m, stage_label, aq_comp_list, sp_comp_list, stoich_aq_dict,
                  ln_k_aq_dict, ln_k_sp_dict, T_stage):
    """Print formatted results for one stage."""
    prec = m.fs.prec
    t0 = m.fs.time.first()

    print("\n" + "=" * 70)
    print(f"{stage_label}  (T = {T_stage} K,  flow_vol = {pyo.value(prec.aqueous_inlet.flow_vol[0]):.1f} L/s)")
    print("=" * 70)

    # ln(K) at process temperature
    print(f"\nln(K) values at T = {T_stage} K (Van't Hoff corrected by model):")
    for r in sorted(prec.merged_rxns):
        lnk = pyo.value(prec.log_k[r])
        print(f"  Rxn {r}: ln(K) = {lnk:+.6f}")

    # Aqueous concentrations
    print("\nAqueous species concentrations:")
    print(f"  {'Species':<15}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in aq_comp_list:
        c_in  = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:>16.4e}  {c_out:>16.4e}")

    # Solid phase
    print("\nSolid phase (mol/s):")
    print(f"  {'Species':<15}  {'Inlet':>16}  {'Outlet':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in sp_comp_list:
        m_in  = pyo.value(prec.cv_precipitate.properties_in[t0].moles_precipitate_comp[sp])
        m_out = pyo.value(prec.cv_precipitate.properties_out[t0].moles_precipitate_comp[sp])
        print(f"  {sp:<15}  {m_in:>16.4e}  {m_out:>16.4e}")

    # Reaction extents
    print("\nReaction extents (mol/L):")
    for r in sorted(prec.merged_rxns):
        xe = pyo.value(prec.rxn_extent[r])
        print(f"  Rxn {r}: {xe:+.4e}")

    # Saturation index for precipitation reactions
    if ln_k_sp_dict:
        print("\nSaturation index check for precipitation reaction(s):")
        for r in sorted(ln_k_sp_dict.keys()):
            log_q = pyo.value(prec.log_q_sp[r])
            lnk   = pyo.value(prec.log_k[r])
            diff  = lnk - log_q
            print(f"  Rxn {r}: ln(Q) = {log_q:.6f},  ln(K) = {lnk:.6f},  "
                  f"ln(K)-ln(Q) = {diff:.2e}  (≈0 at binding)")

    # Equilibrium residual check for aqueous reactions
    print("\nAqueous equilibrium residual check (|Σα·ln(C) − ln(K)| < 1e-6):")
    for r in sorted(ln_k_aq_dict.keys()):
        lhs = sum(
            stoich_aq_dict[r][i] * pyo.value(prec.log_conc_out[i])
            for i in stoich_aq_dict[r]
        )
        residual = abs(lhs - pyo.value(prec.log_k[r]))
        ok = "PASS" if residual < 1e-6 else "FAIL"
        print(f"  Rxn {r}: residual = {residual:.2e}  [{ok}]")

    print("=" * 70)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print("\nBuilding and solving Stage 1 ...")
    m1 = build_stage1_model()
    result1 = solve_model(m1)
    status1 = result1.solver.termination_condition
    print(f"Stage 1 solver status: {status1}")

    if str(status1) == "optimal":
        print_results(
            m1,
            stage_label="Stage 1 — AgCl Precipitation",
            aq_comp_list=AQ_COMP_LIST_S1,
            sp_comp_list=SP_COMP_LIST_S1,
            stoich_aq_dict=STOICH_AQ_DICT_S1,
            ln_k_aq_dict=LN_K_AQ_DICT_S1,
            ln_k_sp_dict=LN_K_SP_DICT_S1,
            T_stage=T_S1,
        )
    else:
        print("WARNING: Stage 1 did not converge — check model formulation.")

    # ------------------------------------------------------------------
    # Stage 2 (uses Stage 1 outlet as inlet — requires Stage 1 solved)
    # ------------------------------------------------------------------
    print("\nBuilding and solving Stage 2 ...")
    m2 = build_stage2_model(m1)
    result2 = solve_model(m2)
    status2 = result2.solver.termination_condition
    print(f"Stage 2 solver status: {status2}")

    if str(status2) == "optimal":
        print_results(
            m2,
            stage_label="Stage 2 — Ag₂C₂O₄ Precipitation",
            aq_comp_list=AQ_COMP_LIST_S2,
            sp_comp_list=SP_COMP_LIST_S2,
            stoich_aq_dict=STOICH_AQ_DICT_S2,
            ln_k_aq_dict=LN_K_AQ_DICT_S2,
            ln_k_sp_dict=LN_K_SP_DICT_S2,
            T_stage=T_S2,
        )
    else:
        print("WARNING: Stage 2 did not converge — check model formulation.")
