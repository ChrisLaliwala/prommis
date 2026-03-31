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

import numpy as np
import pytest
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock

from prommis.opt_precipitator.aqueous_properties import AqueousParameter
from prommis.opt_precipitator.precipitate_properties import PrecipitateParameter
from prommis.opt_precipitator.gas_properties import GasParameter


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
