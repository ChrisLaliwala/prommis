#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Tests for the optimization-based precipitator unit model (opt_precipitator).

Authors: Chris Laliwala
"""

import math

import numpy as np
import pytest
from pyomo.environ import ConcreteModel, SolverFactory, value
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter
from prommis.opt_precipitator.gas_properties import GasParameter
from prommis.opt_precipitator.opt_precipitator import OptPrecipitator


# ---------------------------------------------------------------------------
# Minimal test data shared across property package build tests
# ---------------------------------------------------------------------------

# Two aqueous reactions:
#   rxn 1: Ag+ + Cl- -> AgCl(aq)   log10(K) = 3.31  -> ln(K) = 3.31 * ln(10)
#   rxn 2: H+ + OH- -> H2O         log10(K) = 13.997 -> ln(K) = 13.997 * ln(10)
LN10 = np.log(10)

AQ_COMP_LIST = ["Ag+", "Cl-", "AgCl(aq)", "H+", "OH-"]
LN_K_AQ_DICT = {1: 3.31 * LN10, 2: 13.997 * LN10}
STOICH_AQ_DICT = {
    1: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    2: {"H+": -1, "OH-": -1},
}

# One precipitation reaction:
#   rxn 3: AgCl(s) -> Ag+ + Cl-   log10(Ksp) = -9.75 -> ln(Ksp) = -9.75 * ln(10)
SP_COMP_LIST = ["AgCl(s)"]
LN_K_SP_DICT = {3: -9.75 * LN10}
STOICH_SP_DICT = {
    3: {"Ag+": 1, "Cl-": 1},
}

# One gas reaction:
#   rxn 4: CO2(aq) -> CO2(g)   ln(K) = some value
GAS_COMP_LIST = ["CO2(g)"]
LN_K_GAS_DICT = {4: 1.5}
STOICH_GAS_DICT = {
    4: {"CO2(aq)": -1, "CO2(g)": 1},
}


# ===========================================================================
# Phase 1: Property Package Build Tests
# ===========================================================================


@pytest.mark.build
class TestAqueousParameterBuild:
    @pytest.fixture
    def aq_param(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = AqueousParameter(
            aqueous_comp_list=AQ_COMP_LIST,
            ln_k_aq_dict=LN_K_AQ_DICT,
            stoich_aq_dict=STOICH_AQ_DICT,
        )
        return m.fs.properties

    def test_component_list(self, aq_param):
        assert list(aq_param.component_list) == AQ_COMP_LIST

    def test_rxn_set(self, aq_param):
        assert set(aq_param.rxn_set) == {1, 2}

    def test_ln_k_values(self, aq_param):
        assert aq_param.ln_k_aq_dict[1] == pytest.approx(3.31 * LN10)
        assert aq_param.ln_k_aq_dict[2] == pytest.approx(13.997 * LN10)

    def test_stoich_dict(self, aq_param):
        assert aq_param.stoich_aq_dict[1]["Ag+"] == -1
        assert aq_param.stoich_aq_dict[1]["AgCl(aq)"] == 1

    def test_state_block_class_set(self, aq_param):
        from prommis.opt_precipitator.aqueous_properties import AqueousStateBlock
        assert aq_param.state_block_class is AqueousStateBlock


@pytest.mark.build
class TestPrecipitateParameterBuild:
    @pytest.fixture
    def sp_param(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = PrecipitateParameter(
            precipitate_comp_list=SP_COMP_LIST,
            ln_k_sp_dict=LN_K_SP_DICT,
            stoich_sp_dict=STOICH_SP_DICT,
        )
        return m.fs.properties

    def test_component_list(self, sp_param):
        assert list(sp_param.component_list) == SP_COMP_LIST

    def test_rxn_set(self, sp_param):
        assert set(sp_param.rxn_set) == {3}

    def test_ln_k_values(self, sp_param):
        assert sp_param.ln_k_sp_dict[3] == pytest.approx(-9.75 * LN10)

    def test_stoich_dict(self, sp_param):
        assert sp_param.stoich_sp_dict[3]["Ag+"] == 1
        assert sp_param.stoich_sp_dict[3]["Cl-"] == 1

    def test_state_block_class_set(self, sp_param):
        from prommis.opt_precipitator.precipitate_properties import PrecipitateStateBlock
        assert sp_param.state_block_class is PrecipitateStateBlock


@pytest.mark.build
class TestGasParameterBuild:
    @pytest.fixture
    def gas_param(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = GasParameter(
            gas_comp_list=GAS_COMP_LIST,
            ln_k_gas_dict=LN_K_GAS_DICT,
            stoich_gas_dict=STOICH_GAS_DICT,
            rho_solvent=1000.0,
            MW_solvent=18.015,
        )
        return m.fs.properties

    def test_component_list(self, gas_param):
        assert list(gas_param.component_list) == GAS_COMP_LIST

    def test_rxn_set(self, gas_param):
        assert set(gas_param.rxn_set) == {4}

    def test_ln_k_values(self, gas_param):
        assert gas_param.ln_k_gas_dict[4] == pytest.approx(1.5)

    def test_stoich_dict(self, gas_param):
        assert gas_param.stoich_gas_dict[4]["CO2(aq)"] == -1
        assert gas_param.stoich_gas_dict[4]["CO2(g)"] == 1

    def test_rho_mw_stored(self, gas_param):
        assert gas_param.rho_solvent == pytest.approx(1000.0)
        assert gas_param.MW_solvent == pytest.approx(18.015)

    def test_state_block_class_set(self, gas_param):
        from prommis.opt_precipitator.gas_properties import GasStateBlock
        assert gas_param.state_block_class is GasStateBlock


# ===========================================================================
# Phase 2 Test Data — AgCl precipitation at T=298.15 K, flow_vol=2 L/s
#
# Using standard log10(K) values at 298.15 K (no Van't Hoff correction).
# Analytical check: at saturation, [Ag+] = [Cl-] = sqrt(Ksp(298.15 K))
#   = sqrt(10^{-9.75}) = sqrt(1.78e-10) ≈ 1.334e-5 mol/L
# ===========================================================================

# Reactions:
#   rxn 1: H2O ⇌ H+ + OH-       log10(K) = -13.997
#   rxn 2: Ag+ + Cl- ⇌ AgCl(aq) log10(K) = 3.31
#   rxn 3: AgCl(s) ⇌ Ag+ + Cl-  log10(Ksp) = -9.75
AQ_COMP_LIST_S1 = ["Ag+", "Cl-", "AgCl(aq)", "H+", "OH-"]
SP_COMP_LIST_S1 = ["AgCl(s)"]

LN_K_AQ_DICT_S1 = {1: -13.997 * LN10, 2: 3.31 * LN10}
LN_K_SP_DICT_S1 = {3: -9.75 * LN10}

# stoich_aq_dict covers aqueous species in ALL reactions (aq AND precip)
STOICH_AQ_DICT_S1 = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},
}
STOICH_SP_DICT_S1 = {3: {"AgCl(s)": -1}}


# ===========================================================================
# Phase 2: Unit Model Build Tests
# ===========================================================================


@pytest.mark.build
class TestOptPrecipitatorBuild:
    """Structural tests: verify OptPrecipitator assembles correctly (no solver)."""

    @pytest.fixture
    def model(self):
        m = ConcreteModel()
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
        m.fs.precipitator = OptPrecipitator(
            property_package_aqueous=m.fs.aq_props,
            property_package_precipitate=m.fs.sp_props,
        )
        return m

    def test_control_volumes_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "cv_aqueous")
        assert hasattr(prec, "cv_precipitate")
        assert not hasattr(prec, "cv_gas")

    def test_ports_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "aqueous_inlet")
        assert hasattr(prec, "aqueous_outlet")
        assert hasattr(prec, "precipitate_inlet")
        assert hasattr(prec, "precipitate_outlet")
        assert not hasattr(prec, "gas_inlet")

    def test_decision_variables_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "rxn_extent")
        assert hasattr(prec, "log_conc_out")
        assert hasattr(prec, "log_q_sp")

    def test_rxn_extent_covers_all_reactions(self, model):
        # merged_rxns spans both aqueous (1,2) and precipitation (3) reactions
        assert set(model.fs.precipitator.rxn_extent.keys()) == {1, 2, 3}

    def test_constraints_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "log_conc_linking_eqns")
        assert hasattr(prec, "aqueous_equil_eqns")
        assert hasattr(prec, "log_q_precipitate_equilibrium_rxn_eqns")
        assert hasattr(prec, "precip_sat_ineq")
        assert hasattr(prec, "aqueous_mole_balance_eqns")
        assert hasattr(prec, "vol_balance")
        assert hasattr(prec, "precipitate_mole_balance_eqns")

    def test_objective_exists(self, model):
        assert hasattr(model.fs.precipitator, "min_saturation_index")


# ===========================================================================
# Phase 2: Precipitation Solver Tests
# ===========================================================================


@pytest.mark.solver
class TestPrecipitationSolver:
    """
    Solver tests for AgCl precipitation at T=298.15 K, flow_vol=2 L/s.

    Analytical reference (neglecting complexation):
        [Ag+]* = [Cl-]* = sqrt(Ksp) = sqrt(10^{-9.75}) ≈ 1.334e-5 mol/L
    """

    @pytest.fixture
    def solved_model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.aq_props = AqueousParameter(
            aqueous_comp_list=AQ_COMP_LIST_S1,
            ln_k_aq_dict=LN_K_AQ_DICT_S1,
            stoich_aq_dict=STOICH_AQ_DICT_S1,
        )
        m.fs.sp_props = PrecipitateParameter(
            precipitate_comp_list=SP_COMP_LIST_S1,
            ln_k_sp_dict=LN_K_SP_DICT_S1,
            stoich_sp_dict=STOICH_SP_DICT_S1,
        )
        m.fs.precipitator = OptPrecipitator(
            property_package_aqueous=m.fs.aq_props,
            property_package_precipitate=m.fs.sp_props,
        )
        prec = m.fs.precipitator

        # Fix aqueous inlet (flow_vol in L/s, conc in mol/L)
        prec.aqueous_inlet.flow_vol[0].fix(2.0)
        prec.aqueous_inlet.conc_mol_comp[0, "Ag+"].fix(1e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "Cl-"].fix(1e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "AgCl(aq)"].fix(1e-20)
        prec.aqueous_inlet.conc_mol_comp[0, "H+"].fix(1e-7)
        prec.aqueous_inlet.conc_mol_comp[0, "OH-"].fix(1e-7)

        # Fix precipitate inlet
        prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(1e-5)

        # Initial guesses: start from inlet concentrations
        for comp, c0 in [
            ("Ag+", 1e-4),
            ("Cl-", 1e-4),
            ("AgCl(aq)", 1e-20),
            ("H+", 1e-7),
            ("OH-", 1e-7),
        ]:
            prec.cv_aqueous.properties_out[0].conc_mol_comp[comp].set_value(c0)
            prec.log_conc_out[comp].set_value(math.log(c0))
        prec.cv_aqueous.properties_out[0].flow_vol.set_value(2.0)
        prec.cv_precipitate.properties_out[0].moles_precipitate_comp["AgCl(s)"].set_value(
            1e-5
        )

        solver = SolverFactory("ipopt")
        solver.options = {
            "nlp_scaling_method": "user-scaling",
            "tol": 1e-8,
            "max_iter": 10000,
        }
        results = solver.solve(m, tee=False)
        m._solver_results = results
        return m

    def test_solver_converged(self, solved_model):
        from pyomo.opt import TerminationCondition

        assert (
            solved_model._solver_results.solver.termination_condition
            == TerminationCondition.optimal
        )

    def test_final_ag_concentration(self, solved_model):
        prec = solved_model.fs.precipitator
        ag_out = value(prec.cv_aqueous.properties_out[0].conc_mol_comp["Ag+"])
        # Analytical: sqrt(Ksp(298.15 K)) = sqrt(10^{-9.75}) ≈ 1.334e-5 mol/L
        assert ag_out == pytest.approx(1.334e-5, rel=0.05)

    def test_saturation_index_binding(self, solved_model):
        prec = solved_model.fs.precipitator
        residual = abs(value(prec.log_k[3]) - value(prec.log_q_sp[3]))
        assert residual < 1e-4

    def test_aqueous_equilibrium_satisfied(self, solved_model):
        prec = solved_model.fs.precipitator
        # Rxn 1: H2O ⇌ H+ + OH-
        log_q_rxn1 = value(prec.log_conc_out["H+"]) + value(prec.log_conc_out["OH-"])
        assert log_q_rxn1 == pytest.approx(value(prec.log_k[1]), abs=1e-6)
        # Rxn 2: Ag+ + Cl- ⇌ AgCl(aq)
        log_q_rxn2 = (
            -value(prec.log_conc_out["Ag+"])
            - value(prec.log_conc_out["Cl-"])
            + value(prec.log_conc_out["AgCl(aq)"])
        )
        assert log_q_rxn2 == pytest.approx(value(prec.log_k[2]), abs=1e-6)

    def test_silver_mass_balance(self, solved_model):
        prec = solved_model.fs.precipitator
        flow_vol = value(prec.cv_aqueous.properties_in[0].flow_vol)
        # Total silver (mol/s): Ag+(aq) + AgCl(aq) + AgCl(s)
        ag_in = (
            value(prec.cv_aqueous.properties_in[0].conc_mol_comp["Ag+"]) * flow_vol
            + value(prec.cv_aqueous.properties_in[0].conc_mol_comp["AgCl(aq)"]) * flow_vol
            + value(prec.cv_precipitate.properties_in[0].moles_precipitate_comp["AgCl(s)"])
        )
        ag_out = (
            value(prec.cv_aqueous.properties_out[0].conc_mol_comp["Ag+"]) * flow_vol
            + value(prec.cv_aqueous.properties_out[0].conc_mol_comp["AgCl(aq)"]) * flow_vol
            + value(prec.cv_precipitate.properties_out[0].moles_precipitate_comp["AgCl(s)"])
        )
        assert ag_in == pytest.approx(ag_out, rel=1e-4)


# ===========================================================================
# Phase 3 Test Data — CO₂ carbonic acid system at T=298.15 K, flow_vol=1 L/s
#
# Henry's constant for CO₂ at 298.15 K: H = 1657.7 bar (mole-fraction based).
# The gas-liquid equilibrium in our model reduces to ln(K_gas) = ln(H).
#
# Reactions:
#   rxn 1: H₂O ⇌ H⁺ + OH⁻               log10(K) = -13.997
#   rxn 2: CO₂(aq)+H₂O ⇌ HCO₃⁻+H⁺       log10(K1) = -6.35
#   rxn 3: HCO₃⁻ ⇌ CO₃²⁻ + H⁺           log10(K2) = -10.33
#   rxn 4: CO₂(aq) ⇌ CO₂(g)             ln(K) = ln(1657.7) = 7.413
#
# Analytical equilibrium at P_CO2 = 1.38 bar (used as inlet = outlet):
#   [CO2(aq)] = 0.04619 mol/L
#   [HCO3-]   = [H+] = 1.436e-4 mol/L
#   [CO3(2-)] = 4.677e-11 mol/L
#   [OH-]     = 7.009e-11 mol/L
#   n_CO2(g)  = 0.05568 mol/s
# ===========================================================================

AQ_COMP_LIST_CO2 = ["CO2(aq)", "HCO3-", "CO3(2-)", "H+", "OH-"]
GAS_COMP_LIST_CO2 = ["CO2(g)"]

LN_K_AQ_DICT_CO2 = {
    1: -13.997 * LN10,
    2: -6.35 * LN10,
    3: -10.33 * LN10,
}
LN_K_GAS_DICT_CO2 = {4: math.log(1657.7)}

# stoich_aq_dict covers aqueous species in ALL reactions (including gas rxn 4)
STOICH_AQ_DICT_CO2 = {
    1: {"H+": 1, "OH-": 1},
    2: {"CO2(aq)": -1, "HCO3-": 1, "H+": 1},
    3: {"HCO3-": -1, "CO3(2-)": 1, "H+": 1},
    4: {"CO2(aq)": -1},
}
STOICH_GAS_DICT_CO2 = {4: {"CO2(aq)": -1, "CO2(g)": 1}}


# ===========================================================================
# Phase 3: Gas Extension Build Tests
# ===========================================================================


@pytest.mark.build
class TestOptPrecipitatorGasBuild:
    """Structural tests: verify OptPrecipitator builds correctly with a gas package."""

    @pytest.fixture
    def model(self):
        m = ConcreteModel()
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
            rho_solvent=1000.0,
            MW_solvent=18.015,
        )
        m.fs.precipitator = OptPrecipitator(
            property_package_aqueous=m.fs.aq_props,
            property_package_precipitate=m.fs.sp_props,
            property_package_gas=m.fs.gas_props,
        )
        return m

    def test_gas_control_volume_exists(self, model):
        assert hasattr(model.fs.precipitator, "cv_gas")

    def test_gas_ports_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "gas_inlet")
        assert hasattr(prec, "gas_outlet")

    def test_gas_variables_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "partial_pressure")
        assert hasattr(prec, "log_partial_pressure")
        assert hasattr(prec, "log_moles_gas_out")

    def test_gas_constraints_exist(self, model):
        prec = model.fs.precipitator
        assert hasattr(prec, "log_pressure_linking_eqns")
        assert hasattr(prec, "log_moles_gas_linking_eqns")
        assert hasattr(prec, "gas_equil_eqns")
        assert hasattr(prec, "ideal_gas_eqns")
        assert hasattr(prec, "gas_mole_balance_eqns")

    def test_rxn_extent_includes_gas_reactions(self, model):
        # All four reactions (aq: 1,2; sp: 3; gas: 4) must be in rxn_extent
        assert set(model.fs.precipitator.rxn_extent.keys()) == {1, 2, 3, 4}


# ===========================================================================
# Phase 3: Aqueous-Only Solver Tests
# ===========================================================================


@pytest.mark.solver
class TestAqueousOnlySolver:
    """
    Solver tests for the Ag/Cl aqueous equilibrium system (no precipitation,
    no gas) using Phase 1 test data at T=298.15 K.

    Verifies that the aqueous equilibrium constraints are satisfied at the
    solution — this is the primary check for the gas-absent code path.
    """

    @pytest.fixture
    def solved_model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.aq_props = AqueousParameter(
            aqueous_comp_list=AQ_COMP_LIST,
            ln_k_aq_dict=LN_K_AQ_DICT,
            stoich_aq_dict=STOICH_AQ_DICT,
        )
        # One-species precipitate with no reactions — forces the model into
        # the "no precipitation reactions" code path without IDAES empty-list issues.
        m.fs.sp_props = PrecipitateParameter(
            precipitate_comp_list=["AgCl(s)"],
            ln_k_sp_dict={},
            stoich_sp_dict={},
        )
        m.fs.precipitator = OptPrecipitator(
            property_package_aqueous=m.fs.aq_props,
            property_package_precipitate=m.fs.sp_props,
        )
        prec = m.fs.precipitator

        prec.aqueous_inlet.flow_vol[0].fix(1.0)
        prec.aqueous_inlet.conc_mol_comp[0, "Ag+"].fix(1e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "Cl-"].fix(1e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "AgCl(aq)"].fix(1e-20)
        prec.aqueous_inlet.conc_mol_comp[0, "H+"].fix(1e-7)
        prec.aqueous_inlet.conc_mol_comp[0, "OH-"].fix(1e-7)
        prec.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(1e-20)

        for comp, c0 in [
            ("Ag+", 1e-4),
            ("Cl-", 1e-4),
            ("AgCl(aq)", 1e-20),
            ("H+", 1e-7),
            ("OH-", 1e-7),
        ]:
            prec.cv_aqueous.properties_out[0].conc_mol_comp[comp].set_value(c0)
            prec.log_conc_out[comp].set_value(math.log(c0))
        prec.cv_aqueous.properties_out[0].flow_vol.set_value(1.0)
        prec.cv_precipitate.properties_out[0].moles_precipitate_comp["AgCl(s)"].set_value(
            1e-20
        )

        solver = SolverFactory("ipopt")
        solver.options = {
            "nlp_scaling_method": "user-scaling",
            "tol": 1e-8,
            "max_iter": 10000,
        }
        results = solver.solve(m, tee=False)
        m._solver_results = results
        return m

    def test_solver_converged(self, solved_model):
        from pyomo.opt import TerminationCondition

        assert (
            solved_model._solver_results.solver.termination_condition
            == TerminationCondition.optimal
        )

    def test_aqueous_equilibrium_rxn1(self, solved_model):
        prec = solved_model.fs.precipitator
        # Rxn 1: Ag+ + Cl- ⇌ AgCl(aq), stoich {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1}
        log_q = (
            -value(prec.log_conc_out["Ag+"])
            - value(prec.log_conc_out["Cl-"])
            + value(prec.log_conc_out["AgCl(aq)"])
        )
        assert log_q == pytest.approx(value(prec.log_k[1]), abs=1e-6)

    def test_aqueous_equilibrium_rxn2(self, solved_model):
        prec = solved_model.fs.precipitator
        # Rxn 2: H+ + OH- → H2O, stoich {"H+": -1, "OH-": -1}
        log_q = -value(prec.log_conc_out["H+"]) - value(prec.log_conc_out["OH-"])
        assert log_q == pytest.approx(value(prec.log_k[2]), abs=1e-6)


# ===========================================================================
# Phase 3: Gas-Liquid Solver Tests
# ===========================================================================


@pytest.mark.solver
class TestGasLiquidSolver:
    """
    Solver tests for CO₂ gas-liquid equilibrium at T=298.15 K, flow_vol=1 L/s.

    Inlet conditions are set to the known equilibrium state so the solver
    converges immediately and the result is well-defined.

    Analytical reference:
        P_CO2 ≈ 1.38 bar
        [CO2(aq)] ≈ 0.04619 mol/L
    """

    @pytest.fixture
    def solved_model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.aq_props = AqueousParameter(
            aqueous_comp_list=AQ_COMP_LIST_CO2,
            ln_k_aq_dict=LN_K_AQ_DICT_CO2,
            stoich_aq_dict=STOICH_AQ_DICT_CO2,
        )
        # No precipitation in the CO₂ system — one dummy solid to satisfy the
        # IDAES requirement for a non-empty precipitate property package.
        m.fs.sp_props = PrecipitateParameter(
            precipitate_comp_list=["dummy_solid"],
            ln_k_sp_dict={},
            stoich_sp_dict={},
        )
        m.fs.gas_props = GasParameter(
            gas_comp_list=GAS_COMP_LIST_CO2,
            ln_k_gas_dict=LN_K_GAS_DICT_CO2,
            stoich_gas_dict=STOICH_GAS_DICT_CO2,
            rho_solvent=1000.0,
            MW_solvent=18.015,
        )
        m.fs.precipitator = OptPrecipitator(
            property_package_aqueous=m.fs.aq_props,
            property_package_precipitate=m.fs.sp_props,
            property_package_gas=m.fs.gas_props,
            temperature=298.15,
        )
        prec = m.fs.precipitator

        # Inlet set to the analytical equilibrium state for P_CO2 = 1.38 bar
        prec.aqueous_inlet.flow_vol[0].fix(1.0)
        prec.aqueous_inlet.conc_mol_comp[0, "CO2(aq)"].fix(4.619e-2)
        prec.aqueous_inlet.conc_mol_comp[0, "HCO3-"].fix(1.436e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "CO3(2-)"].fix(4.677e-11)
        prec.aqueous_inlet.conc_mol_comp[0, "H+"].fix(1.436e-4)
        prec.aqueous_inlet.conc_mol_comp[0, "OH-"].fix(7.009e-11)
        prec.precipitate_inlet.moles_precipitate_comp[0, "dummy_solid"].fix(1e-20)
        prec.gas_inlet.moles_gas_comp[0, "CO2(g)"].fix(5.568e-2)

        # Initial guesses from equilibrium concentrations
        eq_concs = {
            "CO2(aq)": 4.619e-2,
            "HCO3-": 1.436e-4,
            "CO3(2-)": 4.677e-11,
            "H+": 1.436e-4,
            "OH-": 7.009e-11,
        }
        for comp, c0 in eq_concs.items():
            prec.cv_aqueous.properties_out[0].conc_mol_comp[comp].set_value(c0)
            prec.log_conc_out[comp].set_value(math.log(c0))
        prec.cv_aqueous.properties_out[0].flow_vol.set_value(1.0)
        prec.cv_precipitate.properties_out[0].moles_precipitate_comp["dummy_solid"].set_value(
            1e-20
        )
        prec.cv_gas.properties_out[0].moles_gas_comp["CO2(g)"].set_value(5.568e-2)
        prec.log_moles_gas_out["CO2(g)"].set_value(math.log(5.568e-2))
        prec.log_partial_pressure["CO2(g)"].set_value(math.log(1.38))
        prec.partial_pressure["CO2(g)"].set_value(1.38)

        solver = SolverFactory("ipopt")
        solver.options = {
            "nlp_scaling_method": "user-scaling",
            "tol": 1e-8,
            "max_iter": 10000,
        }
        results = solver.solve(m, tee=False)
        m._solver_results = results
        return m

    def test_solver_converged(self, solved_model):
        from pyomo.opt import TerminationCondition

        assert (
            solved_model._solver_results.solver.termination_condition
            == TerminationCondition.optimal
        )

    def test_partial_pressure_co2(self, solved_model):
        prec = solved_model.fs.precipitator
        p_co2 = value(prec.partial_pressure["CO2(g)"])
        assert p_co2 == pytest.approx(1.38, rel=0.05)

    def test_gas_liquid_equilibrium_satisfied(self, solved_model):
        prec = solved_model.fs.precipitator
        # ln(K4) = log_P + ln(rho/MW) + (-1)*log_conc_CO2_aq
        import math as _math
        rho_over_MW = 1000.0 / 18.015
        log_q = (
            value(prec.log_partial_pressure["CO2(g)"]) + _math.log(rho_over_MW)
            + (-1) * value(prec.log_conc_out["CO2(aq)"])
        )
        assert log_q == pytest.approx(value(prec.log_k[4]), abs=1e-6)

    def test_ideal_gas_law_satisfied(self, solved_model):
        prec = solved_model.fs.precipitator
        import math as _math
        R_gas = 0.08314  # L·bar / mol / K
        T = 298.15
        flow_vol = value(prec.cv_aqueous.properties_in[0].flow_vol)
        expected_rhs = _math.log(R_gas * T / flow_vol)
        lhs = value(prec.log_partial_pressure["CO2(g)"]) - value(
            prec.log_moles_gas_out["CO2(g)"]
        )
        assert lhs == pytest.approx(expected_rhs, abs=1e-6)

    def test_aqueous_equilibrium_satisfied(self, solved_model):
        prec = solved_model.fs.precipitator
        # Rxn 2: CO₂(aq) + H₂O ⇌ HCO₃⁻ + H⁺
        log_q_rxn2 = (
            -value(prec.log_conc_out["CO2(aq)"])
            + value(prec.log_conc_out["HCO3-"])
            + value(prec.log_conc_out["H+"])
        )
        assert log_q_rxn2 == pytest.approx(value(prec.log_k[2]), abs=1e-6)

    def test_co2_mass_balance(self, solved_model):
        prec = solved_model.fs.precipitator
        flow_vol = value(prec.cv_aqueous.properties_in[0].flow_vol)
        co2_in = (
            value(prec.cv_aqueous.properties_in[0].conc_mol_comp["CO2(aq)"]) * flow_vol
            + value(prec.cv_gas.properties_in[0].moles_gas_comp["CO2(g)"])
        )
        co2_out = (
            value(prec.cv_aqueous.properties_out[0].conc_mol_comp["CO2(aq)"]) * flow_vol
            + value(prec.cv_gas.properties_out[0].moles_gas_comp["CO2(g)"])
        )
        assert co2_in == pytest.approx(co2_out, rel=1e-4)
