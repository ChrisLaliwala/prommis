#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
import math

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    Param,
    TransformationFactory,
    Var,
    assert_optimal_termination,
    value,
)
from pyomo.mpec import Complementarity
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util import DiagnosticsToolbox
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_total_constraints,
    number_unused_variables,
    number_variables,
)

import pytest

from prommis.cmu_precipitator import aqueous_properties as aqueous_thermo_prop_pack
from prommis.cmu_precipitator import gas_properties as gas_thermo_prop_pack
from prommis.cmu_precipitator import (
    precipitate_properties as precipitate_thermo_prop_pack,
)
from prommis.cmu_precipitator.opt_based_precipitator import (
    Precipitator,
    PrecipitatorScaler,
)

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver("ipopt_v2")

LN10 = math.log(10.0)


def assert_unit_units_consistent(unit):
    """Units check that also covers an untransformed Complementarity."""
    for comp in unit.component_objects(descend_into=True):
        if isinstance(comp, Complementarity):
            for cdata in comp.values():
                for arg in cdata._args:
                    assert_units_consistent(arg)
        elif isinstance(comp, (Var, Constraint, Expression, Objective, Param)):
            assert_units_consistent(comp)


# -----------------------------------------------------------------------------
# Ferric hydroxide precipitation from a nitrate solution (the original test case)
def build_feoh3(formulation):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Define aqueous species present in system
    aqueous_comp_list = ["HNO3", "H^+", "OH^-", "NO3^-", "Fe^3+"]
    # Define precipitate species present in system
    precipitate_comp_list = ["FeOH3"]

    # define aqueous equilibrium constants
    aqueous_log_keq_dict = {
        "E1": 1.084,
        "E2": -14,
    }

    # define equilibrium constant for precipitation/dissolution reactions
    precipitate_log_keq_dict = {
        "E3": -33.498,
    }

    # define reaction stoichiometry for aqueous components
    aqueous_stoich_dict = {
        "E1": {"HNO3": -1, "H^+": 1, "NO3^-": 1, "OH^-": 0, "Fe^3+": 0},
        "E2": {"H^+": 1, "OH^-": 1, "HNO3": 0, "NO3^-": 0, "Fe^3+": 0},
        "E3": {"OH^-": 3, "Fe^3+": 1, "HNO3": 0, "NO3^-": 0, "H^+": 0},
    }

    # define reaction stoichiometry for precipitates
    precipitate_stoich_dict = {
        "E1": {"FeOH3": 0},
        "E2": {"FeOH3": 0},
        "E3": {"FeOH3": -1},
    }

    m.fs.aqueous_properties = aqueous_thermo_prop_pack.AqueousParameter(
        aqueous_comp_list=aqueous_comp_list,
        logkeq_dict=aqueous_log_keq_dict,
        stoich_dict=aqueous_stoich_dict,
    )
    m.fs.precipitate_properties = precipitate_thermo_prop_pack.PrecipitateParameter(
        precipitate_comp_list=precipitate_comp_list,
        logkeq_dict=precipitate_log_keq_dict,
        stoich_dict=precipitate_stoich_dict,
    )

    m.fs.unit = Precipitator(
        property_package_aqueous=m.fs.aqueous_properties,
        property_package_precipitate=m.fs.precipitate_properties,
        formulation=formulation,
    )

    # initial concentrations (mol/L) at a flow of 1 L/s
    pH = 7
    conc_in = {
        "HNO3": 1e-10,
        "H^+": 10 ** (-pH),
        "OH^-": 10 ** (-14 + pH),
        "NO3^-": 1.8e-1,
        "Fe^3+": 1.05e-1,
    }
    flow_vol = 1.0

    m.fs.unit.temperature.fix(298.15)
    m.fs.unit.aqueous_inlet.flow_vol[0].fix(flow_vol)
    for j, c in conc_in.items():
        m.fs.unit.aqueous_inlet.flow_mol_comp[0, j].fix(c * flow_vol)
    m.fs.unit.precipitate_inlet.moles_precipitate_comp[0, "FeOH3"].fix(1e-10)

    PrecipitatorScaler().scale_model(m.fs.unit)
    return m


FEOH3_SOLUTION = {  # outlet concentrations (mol/L) and precipitate flow (mol/s)
    "Fe^3+": 0.104766,
    "HNO3": 1.025171e-5,
    "H^+": 0.000691,
    "NO3^-": 0.1799897,
    "OH^-": 1.446944e-11,
    "FeOH3": 0.00023372,
}


def assert_feoh3_solution(m):
    for j in ("Fe^3+", "HNO3", "H^+", "NO3^-", "OH^-"):
        assert pytest.approx(FEOH3_SOLUTION[j], abs=1e-5) == value(
            m.fs.unit.cv_aqueous.properties_out[0].conc_mol_comp[j]
        )
    assert pytest.approx(FEOH3_SOLUTION["FeOH3"], abs=1e-5) == value(
        m.fs.unit.precipitate_outlet.moles_precipitate_comp[0, "FeOH3"]
    )


class TestPrecObjective(object):
    @pytest.fixture(scope="class")
    def prec(self):
        return build_feoh3("objective")

    @pytest.mark.unit
    def test_config(self, prec):
        assert len(prec.fs.unit.config()) == 11

        assert not prec.fs.unit.config.dynamic
        assert not prec.fs.unit.config.has_holdup
        assert prec.fs.unit.config.formulation == "objective"
        assert prec.fs.unit.config.temperature == 298.15
        assert prec.fs.unit.config.gas_volume is None
        assert prec.fs.unit.config.property_package_gas is None
        assert (
            prec.fs.unit.config.property_package_aqueous is prec.fs.aqueous_properties
        )
        assert (
            prec.fs.unit.config.property_package_precipitate
            is prec.fs.precipitate_properties
        )

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, prec):
        assert hasattr(prec.fs.unit, "aqueous_inlet")
        assert len(prec.fs.unit.aqueous_inlet.vars) == 2
        assert hasattr(prec.fs.unit.aqueous_inlet, "flow_vol")
        assert hasattr(prec.fs.unit.aqueous_inlet, "flow_mol_comp")

        assert hasattr(prec.fs.unit, "aqueous_outlet")
        assert len(prec.fs.unit.aqueous_outlet.vars) == 2

        assert hasattr(prec.fs.unit, "precipitate_inlet")
        assert len(prec.fs.unit.precipitate_inlet.vars) == 1
        assert hasattr(prec.fs.unit.precipitate_inlet, "moles_precipitate_comp")

        assert hasattr(prec.fs.unit, "precipitate_outlet")
        assert len(prec.fs.unit.precipitate_outlet.vars) == 1

        assert not hasattr(prec.fs.unit, "gas_inlet")
        assert not hasattr(prec.fs.unit, "cv_gas")

        assert hasattr(prec.fs.unit, "temperature")
        assert hasattr(prec.fs.unit, "inv_temp")
        assert hasattr(prec.fs.unit, "inv_temp_definition")
        assert hasattr(prec.fs.unit, "log_k")
        assert hasattr(prec.fs.unit, "log_conc_out")
        assert hasattr(prec.fs.unit, "log_conc_linking_eqns")
        assert hasattr(prec.fs.unit, "aqueous_equilibrium_eqns")
        assert hasattr(prec.fs.unit, "log_q_sp")
        assert hasattr(prec.fs.unit, "log_q_precipitate_equilibrium_rxn_eqns")
        assert hasattr(prec.fs.unit, "precip_sat_ineq")
        assert hasattr(prec.fs.unit, "min_logs")
        assert not hasattr(prec.fs.unit, "precipitation_complementarity")
        assert hasattr(prec.fs.unit, "aqueous_mole_balance_eqns")
        assert hasattr(prec.fs.unit, "precipitate_mole_balance_eqns")
        assert hasattr(prec.fs.unit, "vol_balance")
        assert hasattr(prec.fs.unit, "volume")

        assert set(prec.fs.unit.merged_rxns) == {"E1", "E2", "E3"}
        assert number_variables(prec.fs.unit) == 25
        assert number_total_constraints(prec.fs.unit) == 17
        assert number_unused_variables(prec.fs.unit) == 0

    @pytest.mark.component
    def test_units(self, prec):
        assert_unit_units_consistent(prec.fs.unit)

    @pytest.mark.component
    def test_dof(self, prec):
        assert degrees_of_freedom(prec) == 1

    @pytest.mark.unit
    def test_scaling(self, prec):
        scaler = PrecipitatorScaler()
        assert prec.fs.unit.default_scaler is PrecipitatorScaler
        assert scaler.get_scaling_factor(prec.fs.unit.temperature) == pytest.approx(
            1e-2
        )
        assert scaler.get_scaling_factor(
            prec.fs.unit.log_conc_out[0, "Fe^3+"]
        ) == pytest.approx(1e-1)
        assert scaler.get_scaling_factor(
            prec.fs.unit.aqueous_inlet.flow_mol_comp[0, "Fe^3+"]
        ) == pytest.approx(1 / 0.105)
        # outlet flows inherit the inlet magnitude
        assert scaler.get_scaling_factor(
            prec.fs.unit.aqueous_outlet.flow_mol_comp[0, "Fe^3+"]
        ) == pytest.approx(1 / 0.105)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize_and_solve(self, prec):
        initializer = prec.fs.unit.default_initializer()
        initializer.initialize(prec.fs.unit)

        results = solver.solve(prec)
        assert_optimal_termination(results)

        dt = DiagnosticsToolbox(prec)
        dt.assert_no_numerical_warnings()

        # fixing final amount of precipitate to its optimal value to avoid non-zero dof warnings
        optimal_feoh3 = value(
            prec.fs.unit.precipitate_outlet.moles_precipitate_comp[0, "FeOH3"]
        )
        prec.fs.unit.precipitate_outlet.moles_precipitate_comp[0, "FeOH3"].fix(
            optimal_feoh3
        )
        dt.assert_no_structural_warnings()

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, prec):
        assert_feoh3_solution(prec)
        # saturated: ln Q = ln Ksp (the quadratic objective is flat at its optimum)
        assert value(prec.fs.unit.log_q_sp[0, "E3"]) == pytest.approx(
            value(prec.fs.unit.log_k["E3"]), abs=1e-2
        )
        assert value(prec.fs.unit.log_k["E3"]) == pytest.approx(-33.498 * LN10)


class TestPrecComplementarity(object):
    @pytest.fixture(scope="class")
    def prec(self):
        return build_feoh3("complementarity")

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, prec):
        assert prec.fs.unit.config.formulation == "complementarity"
        assert isinstance(prec.fs.unit.precipitation_complementarity, Complementarity)
        assert len(prec.fs.unit.precipitation_complementarity) == 1
        assert not hasattr(prec.fs.unit, "precip_sat_ineq")
        assert not hasattr(prec.fs.unit, "min_logs")
        assert hasattr(prec.fs.unit, "log_q_precipitate_equilibrium_rxn_eqns")

    @pytest.mark.component
    def test_units(self, prec):
        assert_unit_units_consistent(prec.fs.unit)

    @pytest.mark.component
    def test_dof(self, prec):
        # the untransformed complementarity leaves the amount of solid undetermined
        assert degrees_of_freedom(prec) == 1

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize_transform_and_solve(self, prec):
        initializer = prec.fs.unit.default_initializer()
        initializer.initialize(prec.fs.unit)
        # the seed is the aqueous equilibrium without precipitation
        assert value(
            prec.fs.unit.precipitate_outlet.moles_precipitate_comp[0, "FeOH3"]
        ) == pytest.approx(1e-10, rel=1e-3)

        # the user chooses the treatment of the complementarity: Scholtes relaxation here
        TransformationFactory("mpec.simple_nonlinear").apply_to(prec, mpec_bound=1e-8)
        prec.obj = Objective(expr=0)
        assert degrees_of_freedom(prec) == 1

        results = solver.solve(prec)
        assert_optimal_termination(results)
        assert_feoh3_solution(prec)


# -----------------------------------------------------------------------------
# Silver chloride: aqueous complexation, precipitation and the Van't Hoff correction
AG_AQ_COMP_LIST = ["H+", "OH-", "Ag+", "Cl-", "AgCl(aq)"]
AG_AQ_LOGK = {1: -13.997, 2: 3.31}  # H2O = H+ + OH-; Ag+ + Cl- = AgCl(aq)
AG_AQ_STOICH = {
    1: {"H+": 1, "OH-": 1},
    2: {"Ag+": -1, "Cl-": -1, "AgCl(aq)": 1},
    3: {"Ag+": 1, "Cl-": 1},  # AgCl(s) = Ag+ + Cl-
}
AG_AQ_DHR = {1: 55_800.0, 2: -12_000.0}
AG_SP_COMP_LIST = ["AgCl(s)"]
AG_SP_LOGK = {3: -9.75}
AG_SP_STOICH = {3: {"AgCl(s)": -1}}
AG_SP_DHR = {3: 65_200.0}
AG_CONC_IN = {"H+": 1e-5, "OH-": 1e-9, "Ag+": 1e-4, "Cl-": 1e-4, "AgCl(aq)": 1e-30}


def build_agcl(temperature, with_solid=True):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.aqueous_properties = aqueous_thermo_prop_pack.AqueousParameter(
        aqueous_comp_list=AG_AQ_COMP_LIST,
        logkeq_dict=AG_AQ_LOGK,
        stoich_dict=(
            AG_AQ_STOICH if with_solid else {k: AG_AQ_STOICH[k] for k in (1, 2)}
        ),
        dHr_dict=AG_AQ_DHR,
    )
    kwargs = {}
    if with_solid:
        m.fs.precipitate_properties = precipitate_thermo_prop_pack.PrecipitateParameter(
            precipitate_comp_list=AG_SP_COMP_LIST,
            logkeq_dict=AG_SP_LOGK,
            stoich_dict=AG_SP_STOICH,
            dHr_dict=AG_SP_DHR,
        )
        kwargs["property_package_precipitate"] = m.fs.precipitate_properties
    m.fs.unit = Precipitator(
        property_package_aqueous=m.fs.aqueous_properties,
        temperature=temperature,
        **kwargs,
    )
    flow_vol = 2.0 if with_solid else 1.0
    m.fs.unit.temperature.fix(temperature)
    m.fs.unit.aqueous_inlet.flow_vol[0].fix(flow_vol)
    for j, c in AG_CONC_IN.items():
        m.fs.unit.aqueous_inlet.flow_mol_comp[0, j].fix(c * flow_vol)
    if with_solid:
        m.fs.unit.precipitate_inlet.moles_precipitate_comp[0, "AgCl(s)"].fix(1e-5)
    PrecipitatorScaler().scale_model(m.fs.unit)
    return m


class TestVantHoff(object):
    @pytest.mark.unit
    def test_log_k_at_temperature(self):
        m = build_agcl(320.0)
        m.fs.unit.inv_temp.set_value(1 / 320.0)
        for r, (logk, dhr) in {
            1: (-13.997, 55_800.0),
            2: (3.31, -12_000.0),
            3: (-9.75, 65_200.0),
        }.items():
            expected = logk * LN10 + dhr / 8.314 * (1 / 298.15 - 1 / 320.0)
            assert value(m.fs.unit.log_k[r]) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_precipitation_298(self):
        m = build_agcl(298.15)
        m.fs.unit.default_initializer().initialize(m.fs.unit)
        assert_optimal_termination(solver.solve(m))
        # saturation: [Ag+] = [Cl-] = sqrt(Ksp), neglecting the complex
        ag = value(m.fs.unit.cv_aqueous.properties_out[0].conc_mol_comp["Ag+"])
        assert ag == pytest.approx(math.sqrt(10**-9.75), rel=0.05)
        assert value(m.fs.unit.log_q_sp[0, 3]) == pytest.approx(
            value(m.fs.unit.log_k[3]), abs=1e-2
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_precipitation_320(self):
        m = build_agcl(320.0)
        m.fs.unit.default_initializer().initialize(m.fs.unit)
        assert_optimal_termination(solver.solve(m))
        # reference value of the original Pyomo model at 320 K
        ag = value(m.fs.unit.cv_aqueous.properties_out[0].conc_mol_comp["Ag+"])
        assert ag == pytest.approx(3.273e-5, rel=1e-3)
        # silver balance
        fv = value(m.fs.unit.cv_aqueous.properties_in[0].flow_vol)
        ag_in = (AG_CONC_IN["Ag+"] + AG_CONC_IN["AgCl(aq)"]) * fv + 1e-5
        ag_out = (
            value(m.fs.unit.aqueous_outlet.flow_mol_comp[0, "Ag+"])
            + value(m.fs.unit.aqueous_outlet.flow_mol_comp[0, "AgCl(aq)"])
            + value(m.fs.unit.precipitate_outlet.moles_precipitate_comp[0, "AgCl(s)"])
        )
        assert ag_in == pytest.approx(ag_out, rel=1e-6)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_all_aqueous_320(self):
        m = build_agcl(320.0, with_solid=False)
        assert not hasattr(m.fs.unit, "precipitate_inlet")
        assert not hasattr(m.fs.unit, "log_q_sp")
        assert not hasattr(m.fs.unit, "min_logs")
        assert degrees_of_freedom(m) == 0
        m.fs.unit.default_initializer().initialize(m.fs.unit)
        m.obj = Objective(expr=0)
        assert_optimal_termination(solver.solve(m))
        conc = m.fs.unit.cv_aqueous.properties_out[0].conc_mol_comp
        assert value(conc["Ag+"]) == pytest.approx(8.851e-5, rel=1e-3)
        assert value(conc["AgCl(aq)"]) == pytest.approx(1.149e-5, rel=1e-3)


# -----------------------------------------------------------------------------
# Carbonic acid system with a CO2 gas phase (Henry's law)
CO2_AQ_COMP_LIST = ["CO3(2-)", "H+", "CO2(aq)", "HCO3-", "OH-"]
CO2_AQ_LOGK = {1: -13.997, 2: -16.681 + 10.3354, 3: -10.3354}
CO2_AQ_STOICH = {
    1: {"H+": 1, "OH-": 1},
    2: {"CO2(aq)": -1, "H+": 1, "HCO3-": 1},
    3: {"HCO3-": -1, "H+": 1, "CO3(2-)": 1},
    4: {"CO2(aq)": -1},
}
CO2_GAS_COMP_LIST = ["CO2(g)"]
CO2_GAS_LOGK = {4: 3.245265839}  # CO2(aq) = CO2(g)
CO2_GAS_STOICH = {4: {"CO2(aq)": -1, "CO2(g)": 1}}
CO2_CONC_IN = {
    "CO3(2-)": 4.9439e-11,
    "H+": 2.1582e-4,
    "CO2(aq)": 9.9484e-2,
    "HCO3-": 2.1582e-4,
    "OH-": 4.8019e-11,
}
CO2_SOLUTION = {
    "CO3(2-)": 4.619553e-11,
    "H+": 1.404529e-4,
    "CO2(aq)": 4.371813e-2,
    "HCO3-": 1.404529e-4,
    "OH-": 7.169175e-11,
}


class TestGasLiquid(object):
    @pytest.fixture(scope="class")
    def prec(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.aqueous_properties = aqueous_thermo_prop_pack.AqueousParameter(
            aqueous_comp_list=CO2_AQ_COMP_LIST,
            logkeq_dict=CO2_AQ_LOGK,
            stoich_dict=CO2_AQ_STOICH,
        )
        m.fs.gas_properties = gas_thermo_prop_pack.GasParameter(
            gas_comp_list=CO2_GAS_COMP_LIST,
            logkeq_dict=CO2_GAS_LOGK,
            stoich_dict=CO2_GAS_STOICH,
            MW_solvent=18.0,  # reference solution basis
        )
        m.fs.unit = Precipitator(
            property_package_aqueous=m.fs.aqueous_properties,
            property_package_gas=m.fs.gas_properties,
        )
        flow_vol = 1.0
        m.fs.unit.temperature.fix(298.15)
        m.fs.unit.aqueous_inlet.flow_vol[0].fix(flow_vol)
        for j, c in CO2_CONC_IN.items():
            m.fs.unit.aqueous_inlet.flow_mol_comp[0, j].fix(c * flow_vol)
        m.fs.unit.gas_inlet.moles_gas_comp[0, "CO2(g)"].fix(1e-20)
        PrecipitatorScaler().scale_model(m.fs.unit)
        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, prec):
        assert hasattr(prec.fs.unit, "cv_gas")
        assert hasattr(prec.fs.unit, "gas_inlet")
        assert hasattr(prec.fs.unit, "gas_outlet")
        assert not hasattr(prec.fs.unit, "cv_precipitate")
        for name in (
            "partial_pressure",
            "log_partial_pressure",
            "log_moles_gas_out",
            "log_pressure_linking_eqns",
            "log_moles_gas_linking_eqns",
            "gas_equilibrium_eqns",
            "ideal_gas_eqns",
            "gas_mole_balance_eqns",
            "gas_volume_basis",
        ):
            assert hasattr(prec.fs.unit, name)
        assert not hasattr(prec.fs.unit, "gas_volume")
        assert set(prec.fs.unit.merged_rxns) == {1, 2, 3, 4}

    @pytest.mark.component
    def test_units(self, prec):
        assert_unit_units_consistent(prec.fs.unit)

    @pytest.mark.component
    def test_dof(self, prec):
        assert degrees_of_freedom(prec) == 0

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, prec):
        prec.fs.unit.default_initializer().initialize(prec.fs.unit)
        prec.obj = Objective(expr=0)
        assert_optimal_termination(solver.solve(prec))

        conc = prec.fs.unit.cv_aqueous.properties_out[0].conc_mol_comp
        for j, c in CO2_SOLUTION.items():
            assert value(conc[j]) == pytest.approx(c, rel=1e-3)
        assert value(prec.fs.unit.partial_pressure[0, "CO2(g)"]) == pytest.approx(
            1.384203, rel=1e-3
        )
        # ideal gas: P V = n R T on the one-second batch basis
        n = value(prec.fs.unit.gas_outlet.moles_gas_comp[0, "CO2(g)"])
        assert value(prec.fs.unit.partial_pressure[0, "CO2(g)"]) == pytest.approx(
            n * 0.08314 * 298.15 / 1.0, rel=1e-6
        )

    @pytest.mark.unit
    def test_gas_volume_option(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.aqueous_properties = aqueous_thermo_prop_pack.AqueousParameter(
            aqueous_comp_list=CO2_AQ_COMP_LIST,
            logkeq_dict=CO2_AQ_LOGK,
            stoich_dict=CO2_AQ_STOICH,
        )
        m.fs.gas_properties = gas_thermo_prop_pack.GasParameter(
            gas_comp_list=CO2_GAS_COMP_LIST,
            logkeq_dict=CO2_GAS_LOGK,
            stoich_dict=CO2_GAS_STOICH,
        )
        m.fs.unit = Precipitator(
            property_package_aqueous=m.fs.aqueous_properties,
            property_package_gas=m.fs.gas_properties,
            gas_volume=5.0,
        )
        assert value(m.fs.unit.gas_volume) == pytest.approx(5.0)
        assert value(m.fs.unit.gas_volume_basis[0]) == pytest.approx(5.0)
        assert_unit_units_consistent(m.fs.unit)
