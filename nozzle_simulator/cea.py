"""RocketCEA integration with explicit SI-oriented units."""

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import RLock

from rocketcea.cea_obj import add_new_fuel, set_rocketcea_data_dir
from rocketcea.cea_obj_w_units import CEA_Obj

if __package__:
    from .models import CEAProperties, NozzleInputs, StationProperties
else:
    # Support direct execution from an editor/terminal while keeping package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nozzle_simulator.models import CEAProperties, NozzleInputs, StationProperties

PARAFFIN_CARD = """
fuel C30H62 C 30 H 62 wt%=83.0
h,cal=-191921.61  t(k)=298.15
fuel C8H8 C 8 H 8 wt%=7.004
h,cal=35444.55  t(k)=298.15
fuel C4H6 C 4 H 6 wt%=4.998
h,cal=26290.63  t(k)=298.15
fuel C3H3N C 3 H 3 N 1 wt%=4.998
h,cal=35156.79  t(k)=298.15
"""

_CEA_LOCK = RLock()


@lru_cache(maxsize=1)
def get_cea() -> CEA_Obj:
    add_new_fuel("PST_Paraffin", PARAFFIN_CARD)
    return CEA_Obj(
        oxName="N2O",
        fuelName="PST_Paraffin",
        pressure_units="Bar",
        temperature_units="K",
        cstar_units="m/s",
        sonic_velocity_units="m/s",
        density_units="kg/m^3",
        specific_heat_units="J/kg-K",
        viscosity_units="poise",
        thermal_cond_units="W/cm-degC",
    )


def _station(temperature, mw_gamma, transport) -> StationProperties:
    molecular_weight, gamma = mw_gamma
    cp, viscosity_poise, conductivity_w_cm_k, prandtl = transport
    raw_values = {
        "temperature": float(temperature),
        "molecular weight": float(molecular_weight),
        "gamma": float(gamma),
        "specific heat": float(cp),
        "viscosity": float(viscosity_poise),
        "thermal conductivity": float(conductivity_w_cm_k),
        "Prandtl number": float(prandtl),
    }
    invalid = [
        name
        for name, value in raw_values.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise ArithmeticError(
            "RocketCEA returned non-positive or non-finite station properties: "
            + ", ".join(invalid)
            + "."
        )
    return StationProperties(
        temperature_k=raw_values["temperature"],
        molecular_weight_g_mol=raw_values["molecular weight"],
        gamma=raw_values["gamma"],
        cp_j_kg_k=raw_values["specific heat"],
        viscosity_pa_s=raw_values["viscosity"] * 0.1,
        conductivity_w_m_k=raw_values["thermal conductivity"] * 100.0,
        prandtl=raw_values["Prandtl number"],
    )


@lru_cache(maxsize=128)
def _invariant_transport_cached(
    pc_bar: float,
    mixture_ratio: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cache chamber/throat transport, which does not depend on exit area ratio."""
    with _CEA_LOCK:
        cea = get_cea()
        reference_eps = 2.0
        chamber = tuple(
            map(
                float,
                cea.get_Chamber_Transport(
                    Pc=pc_bar, MR=mixture_ratio, eps=reference_eps
                ),
            )
        )
        throat = tuple(
            map(
                float,
                cea.get_Throat_Transport(
                    Pc=pc_bar, MR=mixture_ratio, eps=reference_eps
                ),
            )
        )
    return chamber, throat


@lru_cache(maxsize=128)
def _ideal_expansion_ratio_cached(
    pc_bar: float,
    mixture_ratio: float,
    ambient_pressure_bar: float,
) -> float | None:
    if ambient_pressure_bar <= 0.0:
        return None
    with _CEA_LOCK:
        return float(
            get_cea().get_eps_at_PcOvPe(
                Pc=pc_bar,
                MR=mixture_ratio,
                PcOvPe=pc_bar / ambient_pressure_bar,
            )
        )


@lru_cache(maxsize=128)
def _calculate_cached(
    pc_bar: float,
    mixture_ratio: float,
    eps: float,
    ambient_pressure_bar: float,
) -> CEAProperties:
    with _CEA_LOCK:
        cea = get_cea()
        chamber_transport, throat_transport = _invariant_transport_cached(
            pc_bar, mixture_ratio
        )
        temperatures = cea.get_Temperatures(Pc=pc_bar, MR=mixture_ratio, eps=eps)
        chamber = _station(
            temperatures[0],
            cea.get_Chamber_MolWt_gamma(Pc=pc_bar, MR=mixture_ratio, eps=eps),
            chamber_transport,
        )
        throat = _station(
            temperatures[1],
            cea.get_Throat_MolWt_gamma(Pc=pc_bar, MR=mixture_ratio, eps=eps),
            throat_transport,
        )
        exit_station = _station(
            temperatures[2],
            cea.get_exit_MolWt_gamma(Pc=pc_bar, MR=mixture_ratio, eps=eps),
            cea.get_Exit_Transport(Pc=pc_bar, MR=mixture_ratio, eps=eps),
        )
        cf_momentum, cf_ambient, ambient_mode = cea.get_PambCf(
            Pamb=ambient_pressure_bar,
            Pc=pc_bar,
            MR=mixture_ratio,
            eps=eps,
        )
    ideal_expansion_ratio = _ideal_expansion_ratio_cached(
        pc_bar,
        mixture_ratio,
        ambient_pressure_bar,
    )
    return CEAProperties(
        chamber=chamber,
        throat=throat,
        exit=exit_station,
        cstar_m_s=float(cea.get_Cstar(Pc=pc_bar, MR=mixture_ratio)),
        exit_mach=float(cea.get_MachNumber(Pc=pc_bar, MR=mixture_ratio, eps=eps)),
        chamber_to_exit_pressure_ratio=float(
            cea.get_PcOvPe(Pc=pc_bar, MR=mixture_ratio, eps=eps)
        ),
        ideal_momentum_thrust_coefficient=float(cf_momentum),
        ambient_thrust_coefficient=float(cf_ambient),
        ambient_mode=str(ambient_mode),
        ideal_expansion_ratio=ideal_expansion_ratio,
        _chamber_pressure_bar=pc_bar,
    )


def calculate_cea_properties(inputs: NozzleInputs) -> CEAProperties:
    return _calculate_cached(
        round(inputs.chamber_pressure_bar, 8),
        round(inputs.mixture_ratio, 8),
        round(inputs.expansion_ratio, 8),
        round(inputs.ambient_pressure_bar, 8),
    )


def calculate_ideal_expansion_ratio(inputs: NozzleInputs) -> float:
    """Return the CEA area ratio satisfying Pe(epsilon) = Pamb.

    A finite pressure-matched area ratio does not exist for vacuum operation.
    """
    ideal = _ideal_expansion_ratio_cached(
        round(inputs.chamber_pressure_bar, 8),
        round(inputs.mixture_ratio, 8),
        round(inputs.ambient_pressure_bar, 8),
    )
    if ideal is None or not math.isfinite(ideal) or ideal <= 1.0:
        raise ValueError(
            "A finite CEA expansion ratio satisfying Pe = Pamb requires a "
            "positive ambient pressure and epsilon greater than one."
        )
    return float(ideal)


def calculate_exit_pressure_curve(
    chamber_pressure_bar: float,
    mixture_ratio: float,
    expansion_ratios,
):
    return [
        _exit_pressure_cached(
            round(chamber_pressure_bar, 8),
            round(mixture_ratio, 8),
            round(float(eps), 8),
        )
        for eps in expansion_ratios
    ]


@lru_cache(maxsize=2048)
def _exit_pressure_cached(
    chamber_pressure_bar: float,
    mixture_ratio: float,
    expansion_ratio: float,
) -> float:
    with _CEA_LOCK:
        cea = get_cea()
        return float(
            chamber_pressure_bar
            / cea.get_PcOvPe(
                Pc=chamber_pressure_bar,
                MR=mixture_ratio,
                eps=expansion_ratio,
            )
        )


def initialize_cea_worker(temp_root: str) -> None:
    """Give each process its own RocketCEA Fortran working files."""
    worker_directory = Path(temp_root) / f"worker-{os.getpid()}"
    worker_directory.mkdir(parents=True, exist_ok=True)
    set_rocketcea_data_dir(str(worker_directory), do_print=False)
    get_cea.cache_clear()
    _ideal_expansion_ratio_cached.cache_clear()
    _invariant_transport_cached.cache_clear()
    _calculate_cached.cache_clear()
    _exit_pressure_cached.cache_clear()


if __name__ == "__main__":
    from nozzle_simulator.app import main

    main()
