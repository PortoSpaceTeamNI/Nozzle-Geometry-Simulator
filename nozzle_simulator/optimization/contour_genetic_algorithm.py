"""PyGAD adapter for contour optimization using the shared simulator model."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from tempfile import TemporaryDirectory
from threading import Event, Lock

import numpy as np
import pygad
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import qmc

from method_of_caracteristics import MOCSettings, analyze_prescribed_nozzle

from ..boundary_layer import compute_boundary_layer, compute_quick_boundary_layer
from ..cea import (
    calculate_cea_properties,
    calculate_ideal_expansion_ratio,
    initialize_cea_worker,
)
from ..flow import compute_flow
from ..geometry import build_geometry
from ..models import MOCResult, NozzleInputs
from ..performance import compute_performance, performance_fitness


@dataclass(frozen=True)
class OptimizationResult:
    expansion_ratio: float
    bell_fraction: float
    theta_in_deg: float
    theta_sub_deg: float
    theta_out_deg: float
    exit_pressure_bar: float
    ambient_mode: str
    fitness: float
    generations_completed: int
    boundary_layer_model: str
    moc_mass_flow_residual: float | None = None
    moc_training_evaluations: int = 0
    moc_result: MOCResult | None = None


@dataclass(frozen=True)
class MOCContourEvaluation:
    """Exact MOC evaluation used to train or validate the GA response surface."""

    fitness: float
    mass_flow_residual: float
    moc_result: MOCResult | None = None


@dataclass(frozen=True)
class OptimizationSettings:
    """User-configurable GA settings; defaults preserve the original setup."""

    bell_fraction_min: float = 0.60
    bell_fraction_max: float = 1.00
    theta_in_min_deg: float = 20.0
    theta_in_max_deg: float = 35.0
    num_generations: int = 300
    population_size: int = 100
    num_parents_mating: int = 20
    keep_elitism: int = 3
    saturation_generations: int = 40
    crossover_probability: float = 0.85
    mutation_percent_high: int = 67
    mutation_percent_low: int = 34
    evaluation_mode: str = "processes"
    parallel_workers: int = 4
    cache_evaluations: bool = True
    boundary_layer_model: str = "blimp"
    moc_training_samples: int = 24
    moc_shortlist_size: int = 6
    moc_refine_candidates: int = 2
    moc_search_axial_stations: int = 120
    moc_search_radial_stations: int = 21
    moc_refine_axial_stations: int = 360
    moc_refine_radial_stations: int = 61
    moc_random_seed: int = 7321

    def validate(self) -> None:
        ranges = (
            ("Bell fraction", self.bell_fraction_min, self.bell_fraction_max),
            ("Initial wall angle", self.theta_in_min_deg, self.theta_in_max_deg),
        )
        for label, lower, upper in ranges:
            if not np.isfinite((lower, upper)).all() or lower >= upper:
                raise ValueError(f"{label}: the minimum must be lower than the maximum.")
        if self.bell_fraction_min <= 0.0:
            raise ValueError("Bell fraction must be positive.")
        if not 0.0 < self.theta_in_min_deg < self.theta_in_max_deg < 90.0:
            raise ValueError("Initial wall-angle limits must lie between 0 and 90 degrees.")
        if self.num_generations < 1:
            raise ValueError("The number of generations must be at least 1.")
        if self.population_size < 4:
            raise ValueError("The population must contain at least 4 individuals.")
        if not 2 <= self.num_parents_mating <= self.population_size:
            raise ValueError("Parents must be between 2 and the population size.")
        if not 0 <= self.keep_elitism < self.population_size:
            raise ValueError("Elitism must be non-negative and lower than the population size.")
        if self.saturation_generations < 1:
            raise ValueError("Saturation generations must be at least 1.")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("Crossover probability must be between 0 and 1.")
        if not 1 <= self.mutation_percent_low <= self.mutation_percent_high <= 100:
            raise ValueError("Adaptive mutation percentages must satisfy 1 <= low <= high <= 100.")
        if self.evaluation_mode not in {"serial", "threads", "processes"}:
            raise ValueError("Evaluation mode must be 'serial', 'threads' or 'processes'.")
        if self.parallel_workers < 1:
            raise ValueError("The number of evaluation workers must be at least 1.")
        if self.boundary_layer_model not in {"blimp", "quick", "moc"}:
            raise ValueError("Optimization model must be 'blimp', 'quick' or 'moc'.")
        if self.moc_training_samples < 4:
            raise ValueError("MOC optimization requires at least four DOE samples.")
        if self.moc_shortlist_size < 1:
            raise ValueError("The MOC validation shortlist must not be empty.")
        if not 1 <= self.moc_refine_candidates <= self.moc_shortlist_size:
            raise ValueError("MOC refined candidates must fit inside the shortlist.")
        if self.moc_search_axial_stations < 20 or self.moc_refine_axial_stations < 20:
            raise ValueError("MOC axial resolutions must contain at least 20 stations.")
        if self.moc_search_radial_stations < 7 or self.moc_refine_radial_stations < 7:
            raise ValueError("MOC radial resolutions must contain at least 7 stations.")


def _design_inputs(base: NozzleInputs, solution) -> NozzleInputs:
    bell_fraction, theta_in_deg = map(float, solution)
    return replace(
        base,
        bell_fraction=bell_fraction,
        theta_in_deg=theta_in_deg,
    )


def evaluate_contour(
    base: NozzleInputs,
    solution,
    boundary_layer_model: str = "blimp",
) -> float:
    """Effective ambient thrust coefficient for an attached-flow candidate."""
    if boundary_layer_model == "moc":
        return evaluate_moc_contour(base, solution).fitness
    if boundary_layer_model not in {"blimp", "quick"}:
        raise ValueError("Boundary-layer model must be 'blimp', 'quick' or 'moc'.")
    try:
        inputs = _design_inputs(base, solution)
        inputs.validate()
        geometry = build_geometry(inputs)
        cea = calculate_cea_properties(inputs)
        if "Separated" in cea.ambient_mode:
            return 0.0
        flow = compute_flow(inputs, geometry, cea)
        boundary_solver = (
            compute_boundary_layer
            if boundary_layer_model == "blimp"
            else compute_quick_boundary_layer
        )
        boundary_layer = boundary_solver(inputs, geometry, flow, cea)
        performance = compute_performance(inputs, geometry, cea, boundary_layer)
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError):
        return 0.0
    return performance_fitness(performance, cea.ambient_mode)


def evaluate_moc_contour(
    base: NozzleInputs,
    solution,
    *,
    axial_stations: int = 120,
    radial_stations: int = 21,
    return_result: bool = False,
) -> MOCContourEvaluation:
    """Resolve one prescribed contour with BLIMP friction and axisymmetric MOC."""
    try:
        inputs = _design_inputs(base, solution)
        inputs.validate()
        geometry = build_geometry(inputs)
        cea = calculate_cea_properties(inputs)
        if "Separated" in cea.ambient_mode:
            return MOCContourEvaluation(0.0, float("inf"))
        flow = compute_flow(inputs, geometry, cea)
        boundary_layer = compute_boundary_layer(inputs, geometry, flow, cea)
        performance = compute_performance(inputs, geometry, cea, boundary_layer)
        moc = analyze_prescribed_nozzle(
            inputs,
            geometry,
            cea,
            friction_thrust_coefficient=performance.friction_thrust_coefficient,
            settings=MOCSettings(
                axial_stations=axial_stations,
                radial_stations=radial_stations,
                initialization="kliegel_levine",
            ),
        )
        fitness = float(moc.friction_corrected_thrust_coefficient)
        if (
            not np.isfinite(fitness)
            or fitness <= 0.0
            or moc.initial_mass_flow_error > 5.0e-3
        ):
            return MOCContourEvaluation(0.0, moc.mass_flow_residual)
        return MOCContourEvaluation(
            fitness,
            float(moc.mass_flow_residual),
            moc if return_result else None,
        )
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError):
        return MOCContourEvaluation(0.0, float("inf"))


def _evaluate_process_task(task) -> tuple[tuple[float, ...], float]:
    """Picklable worker used by the persistent process pool."""
    base, key, boundary_layer_model = task
    return key, evaluate_contour(base, key, boundary_layer_model)


def _evaluate_moc_process_task(
    task,
) -> tuple[tuple[float, ...], MOCContourEvaluation]:
    """Picklable exact-MOC worker used for the DOE and finalist validation."""
    base, key, axial_stations, radial_stations, return_result = task
    return key, evaluate_moc_contour(
        base,
        key,
        axial_stations=axial_stations,
        radial_stations=radial_stations,
        return_result=return_result,
    )


def _design_bounds(settings: OptimizationSettings) -> tuple[np.ndarray, np.ndarray]:
    low = np.array(
        [settings.bell_fraction_min, settings.theta_in_min_deg],
        dtype=float,
    )
    high = np.array(
        [settings.bell_fraction_max, settings.theta_in_max_deg],
        dtype=float,
    )
    return low, high


def _normalized_design(solution, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (np.asarray(solution, dtype=float) - low) / (high - low)


def _moc_training_designs(settings: OptimizationSettings) -> list[tuple[float, ...]]:
    """Space-filling DOE plus all cube corners, which bound interpolation safely."""
    low, high = _design_bounds(settings)
    normalized = [np.asarray(corner, dtype=float) for corner in product((0.0, 1.0), repeat=2)]
    normalized.append(np.full(2, 0.5))
    sampler = qmc.LatinHypercube(d=2, seed=settings.moc_random_seed)
    normalized.extend(sampler.random(settings.moc_training_samples))
    designs: dict[tuple[float, ...], None] = {}
    for point in normalized:
        physical = low + np.asarray(point) * (high - low)
        designs[tuple(map(float, physical))] = None
    return list(designs)


def _candidate_is_geometrically_valid(base: NozzleInputs, key: tuple[float, ...]) -> bool:
    try:
        inputs = _design_inputs(base, key)
        inputs.validate()
        build_geometry(inputs)
    except (ValueError, ArithmeticError):
        return False
    return True


def optimize_geometry(
    base: NozzleInputs,
    settings: OptimizationSettings | None = None,
    progress: Callable[..., None] | None = None,
    cancel_event: Event | None = None,
    status: Callable[[str], None] | None = None,
    num_generations: int | None = None,
    population_size: int | None = None,
) -> OptimizationResult:
    """Optimize bell fraction and theta_in at the CEA pressure-matched epsilon.

    In ``moc`` mode the expensive solver is sampled in a space-filling DOE. A
    piecewise-linear response surface makes the genetic search inexpensive, then
    an exact coarse and refined MOC re-evaluation selects the returned contour.
    """
    configuration = settings or OptimizationSettings()
    if num_generations is not None:
        configuration = replace(configuration, num_generations=num_generations)
    if population_size is not None:
        configuration = replace(
            configuration,
            population_size=population_size,
            num_parents_mating=min(configuration.num_parents_mating, population_size),
            keep_elitism=min(configuration.keep_elitism, population_size - 1),
        )
    configuration.validate()
    ideal_expansion_ratio = calculate_ideal_expansion_ratio(base)
    base = replace(base, expansion_ratio=ideal_expansion_ratio)

    fitness_cache: dict[tuple[float, ...], float] = {}
    cache_lock = Lock()
    candidate_archive: dict[tuple[float, ...], float] = {}
    process_pool: ProcessPoolExecutor | None = None
    thread_pool: ThreadPoolExecutor | None = None
    process_temp_directory: TemporaryDirectory | None = None
    moc_surrogate: LinearNDInterpolator | None = None
    moc_training_results: dict[tuple[float, ...], MOCContourEvaluation] = {}
    moc_training_count = 0
    reference_cea = calculate_cea_properties(base)
    low, high = _design_bounds(configuration)

    def report_status(message: str) -> None:
        if status is not None:
            status(message)

    def map_moc_tasks(
        keys: Iterable[tuple[float, ...]],
        axial_stations: int,
        radial_stations: int,
        return_result: bool,
        stage: str,
    ) -> dict[tuple[float, ...], MOCContourEvaluation]:
        keys = list(keys)
        tasks = [
            (base, key, axial_stations, radial_stations, return_result)
            for key in keys
        ]
        if process_pool is not None:
            mapped = process_pool.map(_evaluate_moc_process_task, tasks)
        elif thread_pool is not None:
            mapped = thread_pool.map(_evaluate_moc_process_task, tasks)
        else:
            mapped = map(_evaluate_moc_process_task, tasks)
        results: dict[tuple[float, ...], MOCContourEvaluation] = {}
        total = len(tasks)
        for completed, (key, evaluation) in enumerate(mapped, start=1):
            results[key] = evaluation
            report_status(f"{stage}: {completed}/{total} exact MOC evaluations")
            if cancel_event is not None and cancel_event.is_set():
                break
        return results

    def fitness_function(ga_instance, solution, solution_idx):
        key = tuple(map(float, solution))
        if configuration.cache_evaluations:
            with cache_lock:
                cached = fitness_cache.get(key)
            if cached is not None:
                return cached
        if configuration.boundary_layer_model == "moc":
            if not _candidate_is_geometrically_valid(base, key):
                fitness = 0.0
            else:
                value = moc_surrogate(_normalized_design(key, low, high))
                fitness = float(np.asarray(value).reshape(-1)[0])
                if not np.isfinite(fitness) or fitness <= 0.0:
                    fitness = 0.0
        else:
            fitness = evaluate_contour(base, solution, configuration.boundary_layer_model)
        if configuration.cache_evaluations:
            with cache_lock:
                fitness_cache[key] = fitness
        return fitness

    def batch_fitness_function(ga_instance, solutions, solution_indices):
        keys = [tuple(map(float, solution)) for solution in solutions]
        fitness_values: list[float | None] = [None] * len(keys)
        missing_keys: list[tuple[float, ...]] = []
        missing_positions: dict[tuple[float, ...], list[int]] = {}
        for position, key in enumerate(keys):
            cached = fitness_cache.get(key) if configuration.cache_evaluations else None
            if cached is not None:
                fitness_values[position] = cached
            else:
                if key not in missing_positions:
                    missing_keys.append(key)
                    missing_positions[key] = []
                missing_positions[key].append(position)
        if missing_keys:
            tasks = ((base, key, configuration.boundary_layer_model) for key in missing_keys)
            for key, fitness in process_pool.map(_evaluate_process_task, tasks):
                if configuration.cache_evaluations:
                    fitness_cache[key] = fitness
                for position in missing_positions[key]:
                    fitness_values[position] = fitness
        return fitness_values

    def on_generation(ga_instance):
        generation_fitness = np.asarray(ga_instance.last_generation_fitness, dtype=float)
        solution, best, _ = ga_instance.best_solution(pop_fitness=generation_fitness)
        best = float(best)
        order = np.argsort(np.nan_to_num(generation_fitness, nan=-np.inf))[-5:]
        for index in order:
            key = tuple(map(float, ga_instance.population[index]))
            candidate_archive[key] = float(generation_fitness[index])
        if progress is not None:
            candidate_inputs = _design_inputs(base, solution)
            candidate_geometry = build_geometry(candidate_inputs)
            progress(
                ga_instance.generations_completed,
                configuration.num_generations,
                best,
                float(np.mean(generation_fitness)),
                float(np.std(generation_fitness)),
                ideal_expansion_ratio,
                float(solution[0]),
                float(solution[1]),
                candidate_geometry.theta_out_deg,
                reference_cea.exit_pressure_bar,
            )
        if cancel_event is not None and cancel_event.is_set():
            return "stop"
        return None

    if configuration.evaluation_mode == "processes":
        process_temp_directory = TemporaryDirectory(prefix="rocket_nozzle_cea_")
        process_pool = ProcessPoolExecutor(
            max_workers=configuration.parallel_workers,
            initializer=initialize_cea_worker,
            initargs=(process_temp_directory.name,),
        )
    elif configuration.evaluation_mode == "threads":
        thread_pool = ThreadPoolExecutor(max_workers=configuration.parallel_workers)

    ga = None
    final_moc: MOCContourEvaluation | None = None
    try:
        if configuration.boundary_layer_model == "moc":
            training_keys = _moc_training_designs(configuration)
            report_status(
                f"MOC design of experiments: {len(training_keys)} prescribed contours"
            )
            moc_training_results = map_moc_tasks(
                training_keys,
                configuration.moc_search_axial_stations,
                configuration.moc_search_radial_stations,
                False,
                "MOC training",
            )
            moc_training_count = len(moc_training_results)
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("MOC optimization cancelled during its training stage.")
            points = np.asarray(
                [_normalized_design(key, low, high) for key in moc_training_results]
            )
            values = np.asarray(
                [evaluation.fitness for evaluation in moc_training_results.values()]
            )
            if np.count_nonzero(values > 0.0) < 3:
                raise RuntimeError(
                    "The MOC DOE found fewer than three valid contours. Narrow the gene ranges."
                )
            moc_surrogate = LinearNDInterpolator(points, values, fill_value=0.0)
            report_status("MOC response surface ready; starting the genetic search")

        use_process_batch = (
            configuration.evaluation_mode == "processes"
            and configuration.boundary_layer_model != "moc"
        )
        ga = pygad.GA(
            num_generations=configuration.num_generations,
            num_parents_mating=configuration.num_parents_mating,
            fitness_func=batch_fitness_function if use_process_batch else fitness_function,
            fitness_batch_size=configuration.population_size if use_process_batch else None,
            sol_per_pop=configuration.population_size,
            gene_space=[
                {"low": configuration.bell_fraction_min, "high": configuration.bell_fraction_max},
                {"low": configuration.theta_in_min_deg, "high": configuration.theta_in_max_deg},
            ],
            gene_type=float,
            num_genes=2,
            mutation_percent_genes=[
                configuration.mutation_percent_high,
                configuration.mutation_percent_low,
            ],
            parent_selection_type="tournament",
            crossover_type="uniform",
            crossover_probability=configuration.crossover_probability,
            mutation_type="adaptive",
            keep_elitism=configuration.keep_elitism,
            stop_criteria=f"saturate_{configuration.saturation_generations}",
            save_solutions=False,
            on_generation=on_generation,
            suppress_warnings=True,
            random_seed=configuration.moc_random_seed,
            parallel_processing=(
                ["thread", configuration.parallel_workers]
                if configuration.evaluation_mode == "threads"
                and configuration.boundary_layer_model != "moc"
                else None
            ),
        )
        ga.run()
        solution, fitness, _ = ga.best_solution()

        if configuration.boundary_layer_model == "moc":
            final_population_fitness = np.asarray(ga.last_generation_fitness, dtype=float)
            for individual, predicted in zip(ga.population, final_population_fitness):
                candidate_archive[tuple(map(float, individual))] = float(predicted)
            for key, evaluation in moc_training_results.items():
                if evaluation.fitness > 0.0:
                    candidate_archive[key] = evaluation.fitness
            ordered_candidates = [
                key
                for key, _ in sorted(
                    candidate_archive.items(), key=lambda item: item[1], reverse=True
                )
                if _candidate_is_geometrically_valid(base, key)
            ][: configuration.moc_shortlist_size]
            report_status("Validating the MOC shortlist with exact coarse solutions")
            coarse = map_moc_tasks(
                ordered_candidates,
                configuration.moc_search_axial_stations,
                configuration.moc_search_radial_stations,
                False,
                "MOC shortlist",
            )
            valid_coarse = sorted(
                ((key, evaluation) for key, evaluation in coarse.items() if evaluation.fitness > 0.0),
                key=lambda item: item[1].fitness,
                reverse=True,
            )
            if not valid_coarse:
                raise RuntimeError("No shortlisted contour passed exact MOC validation.")
            refine_keys = [key for key, _ in valid_coarse[: configuration.moc_refine_candidates]]
            report_status("Refining the best exact-MOC candidates")
            refined = map_moc_tasks(
                refine_keys,
                configuration.moc_refine_axial_stations,
                configuration.moc_refine_radial_stations,
                True,
                "MOC refinement",
            )
            valid_refined = [
                (key, evaluation)
                for key, evaluation in refined.items()
                if evaluation.fitness > 0.0 and evaluation.moc_result is not None
            ]
            if not valid_refined:
                raise RuntimeError("No finalist passed refined MOC validation.")
            solution, final_moc = max(valid_refined, key=lambda item: item[1].fitness)
            fitness = final_moc.fitness
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
        if thread_pool is not None:
            thread_pool.shutdown(wait=True, cancel_futures=True)
        if process_temp_directory is not None:
            process_temp_directory.cleanup()

    if ga is None or fitness <= 0.0:
        raise RuntimeError("The optimizer did not find a valid contour.")
    optimized_inputs = _design_inputs(base, solution)
    optimized_geometry = build_geometry(optimized_inputs)
    optimized_cea = calculate_cea_properties(optimized_inputs)
    return OptimizationResult(
        expansion_ratio=ideal_expansion_ratio,
        bell_fraction=float(solution[0]),
        theta_in_deg=float(solution[1]),
        theta_sub_deg=base.theta_sub_deg,
        theta_out_deg=optimized_geometry.theta_out_deg,
        exit_pressure_bar=optimized_cea.exit_pressure_bar,
        ambient_mode=optimized_cea.ambient_mode,
        fitness=float(fitness),
        generations_completed=ga.generations_completed,
        boundary_layer_model=configuration.boundary_layer_model,
        moc_mass_flow_residual=final_moc.mass_flow_residual if final_moc is not None else None,
        moc_training_evaluations=moc_training_count,
        moc_result=final_moc.moc_result if final_moc is not None else None,
    )
