#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
import math

from pyomo.environ import ConcreteModel, Var, value

from idaes.core import FlowsheetBlock

import pytest

from prommis.cmu_precipitator import aqueous_properties as aqueous_thermo_prop_pack


@pytest.mark.unit
def test_build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Define aqueous species present in system
    aqueous_comp_list = ["HNO3", "H^+", "OH^-", "NO3^-", "Fe^3+"]

    # define aqueous equilibrium constants
    aqueous_log_keq_dict = {
        "E1": 1.084,
        "E2": -14,
    }

    # define reaction stoichiometry for aqueous components
    aqueous_stoich_dict = {
        "E1": {"HNO3": -1, "H^+": 1, "NO3^-": 1, "OH^-": 0, "Fe^3+": 0},
        "E2": {"H^+": 1, "OH^-": 1, "HNO3": 0, "NO3^-": 0, "Fe^3+": 0},
        "E3": {"OH^-": 3, "Fe^3+": 1, "HNO3": 0, "NO3^-": 0, "H^+": 0},
    }

    m.fs.aqueous_properties = aqueous_thermo_prop_pack.AqueousParameter(
        aqueous_comp_list=aqueous_comp_list,
        logkeq_dict=aqueous_log_keq_dict,
        stoich_dict=aqueous_stoich_dict,
        dHr_dict={"E2": 55_800.0},
    )

    assert set(m.fs.aqueous_properties.rxn_set) == {"E1", "E2"}
    assert m.fs.aqueous_properties.ln_k_dict["E1"] == pytest.approx(
        1.084 * math.log(10)
    )
    assert m.fs.aqueous_properties.dHr_dict == {"E2": 55_800.0}

    m.fs.state = m.fs.aqueous_properties.build_state_block(m.fs.time)
    assert len(m.fs.state) == 1

    assert isinstance(m.fs.state[0].flow_mol_comp, Var)
    assert isinstance(m.fs.state[0].flow_vol, Var)

    m.fs.state[0].flow_vol.set_value(2)
    for i in m.fs.aqueous_properties.component_list:
        m.fs.state[0].flow_mol_comp[i].set_value(0.5)
    assert value(m.fs.state[0].conc_mol_comp["H^+"]) == pytest.approx(0.25)

    m.fs.state.fix_initialization_states()

    assert m.fs.state[0].flow_vol.fixed
    for i in m.fs.aqueous_properties.component_list:
        assert m.fs.state[0].flow_mol_comp[i].fixed
