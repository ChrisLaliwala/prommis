#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Two-Stage Precipitator Flowsheet Example (Equation-Oriented)
=============================================================

This example demonstrates a two-stage flowsheet using two OptPrecipitator unit
models connected in a single equation-oriented (EO) model solved in one IPOPT
call. It replicates the chemistry from the Pyomo reference notebook
``multiple_precipitator_process.ipynb`` (TorresCMULab/precipitator-unit-model,
henryslaw branch).

Because ``AqueousStateBlock`` now uses molar flow rate (``flow_mol_comp``, mol/s)
as the primary state variable — with concentration as a derived
``Expression`` (mol/L = flow_mol_comp / flow_vol) — IDAES Arcs and Mixers can
sum molar flows correctly across streams. This example exploits that property:
Stage 1 outlet molar flows are linked to Stage 2 inlet via equality constraints,
and the Stage 2 volumetric flow rate is the sum of Stage 1 outlet + 1 L/s of
fresh oxalic acid feed. Dilution is handled automatically.

Flowsheet overview
------------------
Stage 1 — AgCl precipitation at T = 320 K, flow_vol = 2.0 L/s
    Ag⁺/Cl⁻ in near-neutral water. AgCl precipitates out.

Mixing point
    Stage 1 aqueous outlet (2.0 L/s) + fresh H₂C₂O₄ solution (1.0 L/s)
    → Stage 2 aqueous inlet (3.0 L/s).
    Ag⁺ and other Stage 1 species are diluted by the extra litre.

Stage 2 — Ag₂C₂O₄ precipitation at T = 300 K, flow_vol = 3.0 L/s
    Diluted Stage 1 outlet + oxalic acid equilibrium.

Chemistry
---------
Stage 1 reactions:
    Rxn 1: H₂O ⇌ H⁺ + OH⁻                    log₁₀(K₂₉₈) = −13.997,  ΔHᵣ = +55,800 J/mol
    Rxn 2: Ag⁺ + Cl⁻ ⇌ AgCl(aq)              log₁₀(K₂₉₈) = +3.31,    ΔHᵣ = −12,000 J/mol
    Rxn 3: AgCl(s) ⇌ Ag⁺ + Cl⁻               log₁₀(K₂₉₈) = −9.75,    ΔHᵣ = +65,200 J/mol

Stage 2 additional reactions (rxns 1–3 as above, plus):
    Rxn 4: H₂C₂O₄(aq) ⇌ 2H⁺ + C₂O₄²⁻        log₁₀(K₂₉₈) = −1.119,   ΔHᵣ = +50,000 J/mol
    Rxn 5: Ag₂C₂O₄(s) ⇌ 2Ag⁺ + C₂O₄²⁻       log₁₀(K₂₉₈) = −3.41,    ΔHᵣ = +25,000 J/mol
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

# --------------------------------------------------------------------------
# Stage 1 — T = 320 K, flow_vol = 2.0 L/s
# --------------------------------------------------------------------------

T_S1 = 320.0
FLOW_VOL_S1 = 2.0  # L/s

AQ_COMP_LIST_S1 = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)"]

LN_K_AQ_DICT_S1 = {
    1: -13.997 * LN10,
    2:    3.31  * LN10,
}
STOICH_AQ_DICT_S1 = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},
}
DHR_AQ_DICT_S1 = {1: +55_800, 2: -12_000}

SP_COMP_LIST_S1 = ["AgCl(s)"]
LN_K_SP_DICT_S1 = {3: -9.75 * LN10}
STOICH_SP_DICT_S1 = {3: {"AgCl(s)": -1}}
DHR_SP_DICT_S1 = {3: +65_200}

C0_S1 = {
    "H+":       1e-5,
    "OH-":      1e-9,
    "Ag+":      1e-4,
    "Cl-":      1e-4,
    "AgCl(aq)": 1e-30,
}
M0_AGCLS_S1 = 1e-5  # mol/s

# --------------------------------------------------------------------------
# Fresh oxalic acid feed added at the mixing point
# --------------------------------------------------------------------------

FLOW_VOL_FRESH = 1.0  # L/s of fresh solution added between stages
C_H2C2O4_FRESH = 1e-5  # mol/L (concentration in the fresh feed stream)

# --------------------------------------------------------------------------
# Stage 2 — T = 300 K, flow_vol = FLOW_VOL_S1 + FLOW_VOL_FRESH = 3.0 L/s
# --------------------------------------------------------------------------

T_S2 = 300.0
FLOW_VOL_S2 = FLOW_VOL_S1 + FLOW_VOL_FRESH  # 3.0 L/s

AQ_COMP_LIST_S2 = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)", "H2C2O4(aq)", "C2O4^{2-}"]

LN_K_AQ_DICT_S2 = {
    1: -13.997 * LN10,
    2:    3.31  * LN10,
    4:   -1.119 * LN10,
}
STOICH_AQ_DICT_S2 = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},
    4: {"H2C2O4(aq)": -1, "H+": 2, "C2O4^{2-}": 1},
    5: {"Ag+": 2, "C2O4^{2-}": 1},
}
DHR_AQ_DICT_S2 = {1: +55_800, 2: -12_000, 4: +50_000}

SP_COMP_LIST_S2 = ["AgCl(s)", "Ag2C2O4(s)"]
LN_K_SP_DICT_S2 = {3: -9.75 * LN10, 5: -3.41 * LN10}
STOICH_SP_DICT_S2 = {3: {"AgCl(s)": -1}, 5: {"Ag2C2O4(s)": -1}}
DHR_SP_DICT_S2 = {3: +65_200, 5: +25_000}

M0_AG2C2O4_S2 = 1e-30  # mol/s (trace initial Ag₂C₂O₄)


# ============================================================================
# Build full EO model (both stages in one ConcreteModel)
# ============================================================================


def build_model():
    """
    Build a two-stage EO flowsheet.

    Both OptPrecipitator blocks share a single ConcreteModel and FlowsheetBlock.
    Mixing constraints at the stage boundary link Stage 1 outlet molar flows
    directly to Stage 2 inlet molar flows, and sum the volumetric flow rates.
    A single IPOPT call solves all variables simultaneously.
    """
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # ---- Property packages ----
    m.fs.aq_props_s1 = AqueousParameter(
        aqueous_comp_list=AQ_COMP_LIST_S1,
        ln_k_aq_dict=LN_K_AQ_DICT_S1,
        stoich_aq_dict=STOICH_AQ_DICT_S1,
        dHr_aq_dict=DHR_AQ_DICT_S1,
    )
    m.fs.sp_props_s1 = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST_S1,
        ln_k_sp_dict=LN_K_SP_DICT_S1,
        stoich_sp_dict=STOICH_SP_DICT_S1,
        dHr_sp_dict=DHR_SP_DICT_S1,
    )
    m.fs.aq_props_s2 = AqueousParameter(
        aqueous_comp_list=AQ_COMP_LIST_S2,
        ln_k_aq_dict=LN_K_AQ_DICT_S2,
        stoich_aq_dict=STOICH_AQ_DICT_S2,
        dHr_aq_dict=DHR_AQ_DICT_S2,
    )
    m.fs.sp_props_s2 = PrecipitateParameter(
        precipitate_comp_list=SP_COMP_LIST_S2,
        ln_k_sp_dict=LN_K_SP_DICT_S2,
        stoich_sp_dict=STOICH_SP_DICT_S2,
        dHr_sp_dict=DHR_SP_DICT_S2,
    )

    # ---- Unit models ----
    m.fs.stage1 = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props_s1,
        property_package_precipitate=m.fs.sp_props_s1,
    )
    m.fs.stage2 = OptPrecipitator(
        property_package_aqueous=m.fs.aq_props_s2,
        property_package_precipitate=m.fs.sp_props_s2,
    )

    # ---- Fix Stage 1 inlet ----
    m.fs.stage1.temperature.fix(T_S1)
    m.fs.stage1.aqueous_inlet.flow_vol[0].fix(FLOW_VOL_S1)
    for sp, conc in C0_S1.items():
        m.fs.stage1.aqueous_inlet.flow_mol_comp[0, sp].fix(conc * FLOW_VOL_S1)
    m.fs.stage1.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(M0_AGCLS_S1)

    # ---- Fix Stage 2 temperature ----
    m.fs.stage2.temperature.fix(T_S2)

    # ---- Mixing constraints: Stage 1 outlet → Stage 2 inlet ----
    # These constraints replace fixed values for Stage 2 aqueous inlet variables.
    # Stage 2 flow_vol = Stage 1 flow_vol + fresh feed flow_vol (dilution).
    # Stage 2 molar flows for Stage 1 species = Stage 1 outlet molar flows.
    # New species (H2C2O4, C2O4^{2-}) are fixed from the fresh feed stream.

    t0 = m.fs.time.first()
    s1_aq_out = m.fs.stage1.cv_aqueous.properties_out[t0]
    s2_aq_in = m.fs.stage2.cv_aqueous.properties_in[t0]
    s1_sp_out = m.fs.stage1.cv_precipitate.properties_out[t0]
    s2_sp_in = m.fs.stage2.cv_precipitate.properties_in[t0]

    # Volumetric flow: Stage 2 inlet = Stage 1 outlet + fresh feed
    @m.fs.Constraint(doc="Stage 2 inlet flow_vol = Stage 1 outlet + fresh feed")
    def mix_flow_vol(fs):
        return s2_aq_in.flow_vol == s1_aq_out.flow_vol + FLOW_VOL_FRESH

    # Molar flows for Stage 1 species pass directly to Stage 2 (no conversion needed)
    m.fs.s1_species_set = pyo.Set(initialize=AQ_COMP_LIST_S1)

    @m.fs.Constraint(m.fs.s1_species_set, doc="Molar flow of Stage 1 species into Stage 2")
    def mix_flow_mol_s1(fs, sp):
        return s2_aq_in.flow_mol_comp[sp] == s1_aq_out.flow_mol_comp[sp]

    # New species in Stage 2 are supplied by the fresh feed at fixed molar flow
    # H2C2O4: 1e-5 mol/L × 1.0 L/s = 1e-5 mol/s
    s2_aq_in.flow_mol_comp["H2C2O4(aq)"].fix(C_H2C2O4_FRESH * FLOW_VOL_FRESH)
    s2_aq_in.flow_mol_comp["C2O4^{2-}"].fix(1e-30 * FLOW_VOL_FRESH)

    # Unfix Stage 2 aqueous inlet variables that are now determined by constraints
    s2_aq_in.flow_vol.unfix()
    for sp in AQ_COMP_LIST_S1:
        s2_aq_in.flow_mol_comp[sp].unfix()

    # Solid: AgCl(s) passes from Stage 1 outlet to Stage 2 inlet
    @m.fs.Constraint(doc="AgCl solid passes from Stage 1 to Stage 2")
    def mix_agcls(fs):
        return s2_sp_in.moles_precipitate_comp["AgCl(s)"] == (
            s1_sp_out.moles_precipitate_comp["AgCl(s)"]
        )

    s2_sp_in.moles_precipitate_comp["AgCl(s)"].unfix()
    s2_sp_in.moles_precipitate_comp["Ag2C2O4(s)"].fix(M0_AG2C2O4_S2)

    # ---- Initialise variables ----
    _initialize(m, t0)

    return m


def _initialize(m, t0):
    """Set initial variable values for both stages from inlet concentrations."""
    s1 = m.fs.stage1
    s2 = m.fs.stage2

    # Stage 1: initialise outlet molar flows and log concentrations from inlet
    for sp in AQ_COMP_LIST_S1:
        c0 = max(C0_S1[sp], 1e-20)
        s1.cv_aqueous.properties_out[t0].flow_mol_comp[sp].set_value(c0 * FLOW_VOL_S1)
        s1.log_conc_out[sp].set_value(math.log(c0))
    s1.cv_aqueous.properties_out[t0].flow_vol.set_value(FLOW_VOL_S1)

    # Stage 2: inlet concentrations after dilution (approximate, pre-equilibrium)
    # Stage 1 species are diluted by FLOW_VOL_FRESH / FLOW_VOL_S2
    s2_aq_in = m.fs.stage2.cv_aqueous.properties_in[t0]
    s2_aq_in.flow_vol.set_value(FLOW_VOL_S2)
    for sp in AQ_COMP_LIST_S1:
        c0_s1 = max(C0_S1[sp], 1e-20)
        # Molar flow from Stage 1 (proxy: Stage 1 inlet × FLOW_VOL_S1)
        s2_aq_in.flow_mol_comp[sp].set_value(c0_s1 * FLOW_VOL_S1)

    # Stage 2: initialise outlet molar flows and log concentrations
    for sp in AQ_COMP_LIST_S2:
        if sp in AQ_COMP_LIST_S1:
            c0 = max(C0_S1[sp], 1e-20) * FLOW_VOL_S1 / FLOW_VOL_S2  # diluted
        elif sp == "H2C2O4(aq)":
            c0 = C_H2C2O4_FRESH * FLOW_VOL_FRESH / FLOW_VOL_S2
        else:
            c0 = 1e-20
        s2.cv_aqueous.properties_out[t0].flow_mol_comp[sp].set_value(c0 * FLOW_VOL_S2)
        s2.log_conc_out[sp].set_value(math.log(c0))
    s2.cv_aqueous.properties_out[t0].flow_vol.set_value(FLOW_VOL_S2)


# ============================================================================
# Solve
# ============================================================================


def solve_model(m):
    """Solve both stages simultaneously with IPOPT (single EO solve)."""
    solver = pyo.SolverFactory("ipopt")
    solver.options["nlp_scaling_method"] = "user-scaling"
    solver.options["tol"] = 1e-8
    solver.options["max_iter"] = 10_000
    result = solver.solve(m, tee=False)
    return result


# ============================================================================
# Results
# ============================================================================


def _print_stage_results(prec, stage_label, aq_comp_list, sp_comp_list,
                         stoich_aq_dict, ln_k_aq_dict, ln_k_sp_dict, T_stage):
    """Print results for one stage."""
    t0 = prec.flowsheet().time.first()

    print("\n" + "=" * 70)
    print(f"{stage_label}  (T = {T_stage} K,  "
          f"flow_vol_out = {pyo.value(prec.cv_aqueous.properties_out[t0].flow_vol):.1f} L/s)")
    print("=" * 70)

    print(f"\nln(K) values at T = {T_stage} K (Van't Hoff corrected):")
    for r in sorted(prec.merged_rxns):
        lnk = pyo.value(prec.log_k[r])
        print(f"  Rxn {r}: ln(K) = {lnk:+.6f}")

    print("\nAqueous species concentrations:")
    print(f"  {'Species':<15}  {'Inlet (mol/L)':>16}  {'Outlet (mol/L)':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in aq_comp_list:
        c_in  = pyo.value(prec.cv_aqueous.properties_in[t0].conc_mol_comp[sp])
        c_out = pyo.value(prec.cv_aqueous.properties_out[t0].conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:>16.4e}  {c_out:>16.4e}")

    print("\nSolid phase (mol/s):")
    print(f"  {'Species':<15}  {'Inlet':>16}  {'Outlet':>16}")
    print(f"  {'-'*15}  {'-'*16}  {'-'*16}")
    for sp in sp_comp_list:
        m_in  = pyo.value(prec.cv_precipitate.properties_in[t0].moles_precipitate_comp[sp])
        m_out = pyo.value(prec.cv_precipitate.properties_out[t0].moles_precipitate_comp[sp])
        print(f"  {sp:<15}  {m_in:>16.4e}  {m_out:>16.4e}")

    print("\nReaction extents (mol/L):")
    for r in sorted(prec.merged_rxns):
        xe = pyo.value(prec.rxn_extent[r])
        print(f"  Rxn {r}: {xe:+.4e}")

    if ln_k_sp_dict:
        print("\nSaturation index check for precipitation reaction(s):")
        for r in sorted(ln_k_sp_dict.keys()):
            log_q = pyo.value(prec.log_q_sp[r])
            lnk   = pyo.value(prec.log_k[r])
            diff  = lnk - log_q
            print(f"  Rxn {r}: ln(Q) = {log_q:.6f},  ln(K) = {lnk:.6f},  "
                  f"ln(K)-ln(Q) = {diff:.2e}  (≈0 at binding)")

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


def print_results(m):
    """Print results for both stages."""
    _print_stage_results(
        m.fs.stage1,
        stage_label="Stage 1 — AgCl Precipitation",
        aq_comp_list=AQ_COMP_LIST_S1,
        sp_comp_list=SP_COMP_LIST_S1,
        stoich_aq_dict=STOICH_AQ_DICT_S1,
        ln_k_aq_dict=LN_K_AQ_DICT_S1,
        ln_k_sp_dict=LN_K_SP_DICT_S1,
        T_stage=T_S1,
    )

    # Show diluted inlet concentrations for Stage 2
    t0 = m.fs.time.first()
    s2_in = m.fs.stage2.cv_aqueous.properties_in[t0]
    flow_vol_s2_in = pyo.value(s2_in.flow_vol)
    print(f"\nMixing point: Stage 1 outlet ({FLOW_VOL_S1:.1f} L/s) + fresh feed "
          f"({FLOW_VOL_FRESH:.1f} L/s) → Stage 2 inlet ({flow_vol_s2_in:.1f} L/s)")
    print("Stage 2 inlet concentrations after dilution:")
    for sp in AQ_COMP_LIST_S2:
        c_in = pyo.value(s2_in.conc_mol_comp[sp])
        print(f"  {sp:<15}  {c_in:.4e} mol/L")

    _print_stage_results(
        m.fs.stage2,
        stage_label="Stage 2 — Ag₂C₂O₄ Precipitation",
        aq_comp_list=AQ_COMP_LIST_S2,
        sp_comp_list=SP_COMP_LIST_S2,
        stoich_aq_dict=STOICH_AQ_DICT_S2,
        ln_k_aq_dict=LN_K_AQ_DICT_S2,
        ln_k_sp_dict=LN_K_SP_DICT_S2,
        T_stage=T_S2,
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("Building two-stage EO flowsheet ...")
    m = build_model()

    print("Solving (single EO IPOPT call for both stages) ...")
    result = solve_model(m)

    status = result.solver.termination_condition
    print(f"Solver status: {status}")

    if str(status) == "optimal":
        print_results(m)
    else:
        print("WARNING: solver did not converge — check model formulation.")
