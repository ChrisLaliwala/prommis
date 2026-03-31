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
