"""Desktop interface with embedded 2D, 3D and engineering plots."""

import sys
import time
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from method_of_caracteristics import MOCSettings, analyze_prescribed_nozzle

from .cea import calculate_exit_pressure_curve, calculate_ideal_expansion_ratio
from .export import export_result
from .models import ATM_TO_BAR, NozzleInputs, SimulationResult
from .optimization import OptimizationResult, OptimizationSettings, optimize_geometry
from .performance import loss_breakdown
from .simulation import simulate

FIELD_GROUPS = {
    "Operating point — RocketCEA": [
        ("chamber_pressure_bar", "Chamber pressure", "bar", "30.0"),
        ("mixture_ratio", "Mixture ratio", "O/F", "6.5"),
        ("ambient_pressure_atm", "Ambient pressure", "atm", "1.0"),
        ("expansion_ratio", "Expansion ratio", "Ae/At", "5.6"),
    ],
    "Pre-sizing": [
        ("throat_radius_mm", "Throat radius", "mm", "17.28"),
        ("chamber_diameter_mm", "Chamber diameter", "mm", "120.0"),
    ],
    "Contour design variables": [
        ("bell_fraction_percent", "Bell fraction", "%", "80.0"),
        ("theta_in_deg", "Initial wall angle", "deg", "30.0"),
        ("theta_sub_deg", "Convergent angle", "deg", "50.0"),
    ],
    "Fixed geometry inputs": [
        ("reference_half_angle_deg", "Cone half-angle α", "deg", "15.0"),
    ],
}

OPTIMIZATION_RANGE_FIELDS = (
    ("bell_fraction", "Bell fraction", "%", "60.0", "100.0"),
    ("theta_in", "Initial wall angle", "deg", "20.0", "45.0"),
)

OPTIMIZATION_GA_FIELDS = (
    ("num_generations", "Generations", "300"),
    ("population_size", "Population size", "100"),
    ("num_parents_mating", "Mating parents", "20"),
    ("keep_elitism", "Elite individuals", "3"),
    ("saturation_generations", "Saturation limit", "40"),
    ("crossover_probability", "Crossover probability", "0.85"),
    ("mutation_percent_high", "Adaptive mutation — high", "67"),
    ("mutation_percent_low", "Adaptive mutation — low", "34"),
)

MOC_OPTIMIZATION_RESOLUTIONS = {
    "Fast MOC - 600 x 101": (600, 101),
    "Precise MOC - 1200 x 201": (1200, 201),
    "Study - 140 x 41": (140, 41),
    "Study - 240 x 41": (240, 41),
    "Study - 360 x 61": (360, 61),
    "Study - 480 x 81": (480, 81),
    "720 x 121 - Mesh study": (720, 121),
    "840 x 141 - Mesh study": (840, 141),
    "960 x 161 - Mesh study": (960, 161),
    "1080 x 181 - Mesh study": (1080, 181),
}
DEFAULT_MOC_OPTIMIZATION_RESOLUTION = "Fast MOC - 600 x 101"


class NozzleSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rocket Nozzle Simulator")
        self.minsize(1180, 720)
        self._maximize_window()
        self.result: SimulationResult | None = None
        self.entries: dict[str, ttk.Entry] = {}
        self.optimization_entries: dict[str, ttk.Entry] = {}
        self.moc_entries: dict[str, ttk.Entry] = {}
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self.figures: dict[str, Figure] = {}
        self.tab_frames: dict[str, ttk.Frame] = {}
        self.optimization_queue: Queue = Queue()
        self.optimization_cancel = Event()
        self.optimization_thread: Thread | None = None
        self.optimization_result: OptimizationResult | None = None
        self.moc_queue: Queue = Queue()
        self.moc_thread: Thread | None = None
        self.moc_initialization_var = tk.StringVar(
            value="Kliegel-Levine characteristic net"
        )
        self.active_boundary_layer_model = "blimp"
        self.optimization_started_at: float | None = None
        self._configure_style()
        self._build_layout()
        self.after(150, self.run_simulation)

    def _maximize_window(self):
        """Start maximized while keeping normal window controls available."""
        try:
            self.state("zoomed")  # Windows
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)  # Linux window managers
            except tk.TclError:
                width = self.winfo_screenwidth()
                height = self.winfo_screenheight()
                self.geometry(f"{width}x{height}+0+0")

    def _configure_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 9))
        style.configure("PanelTitle.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("ResultValue.TLabel", font=("Consolas", 10))
        style.configure("Status.TLabel", padding=(8, 5))

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, padding=(18, 12))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Rocket Nozzle Simulator", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="Bell geometry · RocketCEA · Flow · Thermal · Boundary layer",
        ).pack(side="left", padx=18, pady=(7, 0))

        controls = ttk.Frame(self, padding=(12, 0, 8, 8), width=340)
        controls.grid(row=1, column=0, sticky="nsw")
        controls.grid_propagate(False)
        self._build_controls(controls)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=1, sticky="nsew", padx=(0, 12), pady=(0, 8))
        optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(optimization_frame, text="Optimization")
        self.tab_frames["Optimization"] = optimization_frame
        self._build_optimization_tab(optimization_frame)
        moc_frame = ttk.Frame(self.notebook)
        self.notebook.add(moc_frame, text="MOC analysis")
        self.tab_frames["MOC analysis"] = moc_frame
        self._build_moc_tab(moc_frame)
        for name in (
            "Geometry 2D",
            "Interactive 3D",
            "Expansion sizing",
            "Flow profiles",
            "Thermal",
            "Boundary layer",
            "Loss breakdown",
        ):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.tab_frames[name] = frame
            figure = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.update(); toolbar.pack(fill="x")
            self.figures[name] = figure
            self.canvases[name] = canvas

        self.status = ttk.Label(self, text="Ready", style="Status.TLabel", anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._refresh_optimization_fixed_inputs()

    def _build_controls(self, parent):
        canvas = tk.Canvas(parent, width=326, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(2, 2, 8, 8))
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw", width=310)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0
        for group, fields in FIELD_GROUPS.items():
            box = ttk.LabelFrame(content, text=group, style="Section.TLabelframe", padding=9)
            box.grid(row=row, column=0, sticky="ew", pady=(0, 9))
            box.columnconfigure(1, weight=1)
            for index, (key, label, unit, default) in enumerate(fields):
                ttk.Label(box, text=label).grid(row=index, column=0, sticky="w", pady=3)
                entry = ttk.Entry(box, width=11)
                entry.insert(0, default)
                entry.grid(row=index, column=1, sticky="ew", padx=7, pady=3)
                ttk.Label(box, text=unit, width=6).grid(row=index, column=2, sticky="w")
                self.entries[key] = entry
            row += 1

        actions = ttk.Frame(content)
        actions.grid(row=row, column=0, sticky="ew", pady=(2, 10))
        actions.columnconfigure((0, 1), weight=1)
        self.generate_button = ttk.Button(
            actions, text="Generate geometry", style="Primary.TButton", command=self.run_simulation
        )
        self.generate_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.optimize_button = ttk.Button(
            actions,
            text="Optimize geometry",
            style="Primary.TButton",
            command=self.show_optimization_tab,
        )
        self.optimize_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.moc_button = ttk.Button(
            actions,
            text="Run axisymmetric MOC",
            style="Primary.TButton",
            command=self.run_moc_analysis,
        )
        self.moc_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="Export results", command=self.export_results).grid(
            row=3, column=0, sticky="ew", pady=6, padx=(0, 3)
        )
        ttk.Button(actions, text="Reset", command=self.reset_defaults).grid(
            row=3, column=1, sticky="ew", pady=6, padx=(3, 0)
        )
        row += 1

        results_box = ttk.LabelFrame(content, text="Calculated result", padding=9)
        results_box.grid(row=row, column=0, sticky="ew")
        self.summary = tk.Text(
            results_box, width=35, height=18, relief="flat", background="#f3f3f3",
            font=("Consolas", 9), wrap="none",
        )
        self.summary.pack(fill="both", expand=True)
        self.summary.configure(state="disabled")

    def _build_moc_tab(self, parent):
        """Build prescribed-wall MOC controls and field visualizations."""
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(
            parent,
            text="Axisymmetric characteristic marching",
            style="Section.TLabelframe",
            padding=10,
        )
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(7, weight=1)
        ttk.Label(controls, text="Initial-line model").grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        initialization_box = ttk.Combobox(
            controls,
            textvariable=self.moc_initialization_var,
            values=(
                "Kliegel-Levine characteristic net",
                "Quasi-1D reference",
                "Sauer projected (diagnostic only)",
            ),
            state="readonly",
            width=27,
        )
        initialization_box.grid(row=0, column=1, sticky="w", padx=(0, 14))
        fields = (
            ("axial_stations", "Axial stations", "360"),
            ("radial_stations", "Radial stations Nr", "61"),
            ("start_mach", "Quasi-1D reference Mach", "1.12"),
        )
        for index, (key, label, default) in enumerate(fields):
            column = 2 * index + 2
            ttk.Label(controls, text=label).grid(
                row=0, column=column, sticky="w", padx=(0, 5)
            )
            entry = ttk.Entry(controls, width=9)
            entry.insert(0, default)
            entry.grid(row=0, column=column + 1, sticky="w", padx=(0, 14))
            self.moc_entries[key] = entry

        self.moc_tab_button = ttk.Button(
            controls,
            text="Run MOC analysis",
            style="Primary.TButton",
            command=self.run_moc_analysis,
        )
        self.moc_tab_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.moc_progress = ttk.Progressbar(controls, mode="determinate", maximum=360)
        self.moc_progress.grid(
            row=1, column=2, columnspan=6, sticky="ew", padx=(14, 0), pady=(8, 0)
        )
        self.moc_state_var = tk.StringVar(
            value="Generate the geometry, then run the axisymmetric analysis."
        )
        ttk.Label(
            controls,
            textvariable=self.moc_state_var,
            wraplength=1250,
        ).grid(row=2, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        plot_host = ttk.Frame(parent)
        plot_host.grid(row=1, column=0, sticky="nsew")
        figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(figure, master=plot_host)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.figures["MOC analysis"] = figure
        self.canvases["MOC analysis"] = canvas
        self._plot_moc()

    def _build_optimization_tab(self, parent):
        """Build the GA workspace inside the main application window."""
        parent.configure(padding=18)
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(2, weight=1)

        ttk.Label(parent, text="Nozzle contour optimization", style="PanelTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            parent,
            text=(
                "Define the design-space intervals and genetic-algorithm controls here. "
                "Pc, O/F, Pamb, pre-sizing and fixed geometry are taken from the left panel."
            ),
            wraplength=960,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 14))

        settings_host = ttk.Frame(parent)
        settings_host.grid(row=2, column=0, sticky="nsew", padx=(0, 9))
        settings_host.columnconfigure(0, weight=1)
        settings_host.rowconfigure(0, weight=1)
        settings_canvas = tk.Canvas(settings_host, highlightthickness=0)
        settings_scrollbar = ttk.Scrollbar(
            settings_host, orient="vertical", command=settings_canvas.yview
        )
        settings_panel = ttk.Frame(settings_canvas)
        settings_window = settings_canvas.create_window(
            (0, 0), window=settings_panel, anchor="nw"
        )
        settings_panel.bind(
            "<Configure>",
            lambda event: settings_canvas.configure(
                scrollregion=settings_canvas.bbox("all")
            ),
        )
        settings_canvas.bind(
            "<Configure>",
            lambda event: settings_canvas.itemconfigure(
                settings_window, width=event.width
            ),
        )
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar.grid(row=0, column=1, sticky="ns")
        settings_panel.columnconfigure(0, weight=1)

        ranges = ttk.LabelFrame(
            settings_panel, text="Geometry search ranges", style="Section.TLabelframe", padding=12
        )
        ranges.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ranges.columnconfigure((1, 2), weight=1)
        self.optimization_ideal_epsilon_var = tk.StringVar(value="Calculated from Pe = Pamb")
        ttk.Label(ranges, text="CEA-sized expansion ratio").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 8)
        )
        ttk.Label(
            ranges,
            textvariable=self.optimization_ideal_epsilon_var,
            style="ResultValue.TLabel",
        ).grid(row=0, column=1, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(ranges, text="Design variable").grid(row=1, column=0, sticky="w", padx=(0, 12))
        ttk.Label(ranges, text="Minimum").grid(row=1, column=1, sticky="w")
        ttk.Label(ranges, text="Maximum").grid(row=1, column=2, sticky="w")
        ttk.Label(ranges, text="Unit").grid(row=1, column=3, sticky="w")
        for row, (key, label, unit, lower, upper) in enumerate(
            OPTIMIZATION_RANGE_FIELDS, start=2
        ):
            ttk.Label(ranges, text=label).grid(row=row, column=0, sticky="w", pady=5)
            for column, suffix, default in ((1, "min", lower), (2, "max", upper)):
                entry = ttk.Entry(ranges, width=13)
                entry.insert(0, default)
                entry.grid(row=row, column=column, sticky="ew", padx=(0, 9), pady=5)
                self.optimization_entries[f"{key}_{suffix}"] = entry
            ttk.Label(ranges, text=unit).grid(row=row, column=3, sticky="w", pady=5)

        ga_box = ttk.LabelFrame(
            settings_panel,
            text="Genetic algorithm configuration",
            style="Section.TLabelframe",
            padding=12,
        )
        ga_box.grid(row=1, column=0, sticky="ew")
        ga_box.columnconfigure((1, 3), weight=1)
        for index, (key, label, default) in enumerate(OPTIMIZATION_GA_FIELDS):
            row, group = divmod(index, 2)
            label_column = group * 2
            ttk.Label(ga_box, text=label).grid(
                row=row, column=label_column, sticky="w", padx=(0, 8), pady=5
            )
            entry = ttk.Entry(ga_box, width=12)
            entry.insert(0, default)
            entry.grid(
                row=row,
                column=label_column + 1,
                sticky="ew",
                padx=(0, 18 if group == 0 else 0),
                pady=5,
            )
            self.optimization_entries[key] = entry

        ttk.Label(
            ga_box,
            text=(
                "Operators preserved from the GA: tournament selection · uniform crossover · "
                "adaptive mutation. Genes: bell fraction and initial wall angle. Expansion "
                "ratio is sized by CEA from Pe = Pamb; the exit angle is derived. The "
                "convergent angle is fixed by the user. "
                "Objective: effective ambient thrust coefficient."
            ),
            wraplength=680,
        ).grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))

        execution_box = ttk.LabelFrame(
            settings_panel,
            text="Evaluation acceleration",
            style="Section.TLabelframe",
            padding=12,
        )
        execution_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        execution_box.columnconfigure((1, 3), weight=1)
        ttk.Label(execution_box, text="Evaluation mode").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.optimization_evaluation_mode = tk.StringVar(value="Processes")
        self.optimization_mode_combo = ttk.Combobox(
            execution_box,
            textvariable=self.optimization_evaluation_mode,
            values=("Processes", "Serial", "Threads"),
            state="readonly",
            width=12,
        )
        self.optimization_mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 18))
        ttk.Label(execution_box, text="Evaluation workers").grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )
        workers_entry = ttk.Entry(execution_box, width=12)
        workers_entry.insert(0, "4")
        workers_entry.grid(row=0, column=3, sticky="ew")
        self.optimization_entries["parallel_workers"] = workers_entry
        self.optimization_cache_evaluations = tk.BooleanVar(value=True)
        self.optimization_cache_check = ttk.Checkbutton(
            execution_box,
            text="Reuse the fitness of repeated designs (exact-match cache)",
            variable=self.optimization_cache_evaluations,
        )
        self.optimization_cache_check.grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(execution_box, text="Final MOC resolution").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(10, 0)
        )
        self.optimization_moc_resolution = tk.StringVar(
            value=DEFAULT_MOC_OPTIMIZATION_RESOLUTION
        )
        self.optimization_moc_resolution_combo = ttk.Combobox(
            execution_box,
            textvariable=self.optimization_moc_resolution,
            values=tuple(MOC_OPTIMIZATION_RESOLUTIONS),
            state="readonly",
            width=30,
        )
        self.optimization_moc_resolution_combo.grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 0)
        )
        self.optimization_moc_resolution_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self._refresh_optimization_fixed_inputs(),
        )
        ttk.Label(
            execution_box,
            text=(
                "DOE remains 120 x 21; this preset controls exact finalists. "
                "Fast reproduced the Precise winner with Cf +0.348% in this study."
            ),
            wraplength=220,
        ).grid(row=2, column=3, sticky="w", padx=(10, 0), pady=(10, 0))

        history_box = ttk.LabelFrame(
            settings_panel,
            text="Optimization history",
            style="Section.TLabelframe",
            padding=8,
        )
        history_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.optimization_history_figure = Figure(
            figsize=(7.0, 5.4), dpi=100, constrained_layout=True
        )
        self.optimization_history_axes = self.optimization_history_figure.subplots(
            2, 1, sharex=True
        )
        self.optimization_history_canvas = FigureCanvasTkAgg(
            self.optimization_history_figure, master=history_box
        )
        self.optimization_history_canvas.get_tk_widget().pack(
            fill="both", expand=True
        )
        self.optimization_history = {
            "generation": [],
            "best": [],
            "mean": [],
            "std": [],
            "genes": [],
        }
        self._plot_optimization_history()

        run_host = ttk.Frame(parent)
        run_host.grid(row=2, column=1, sticky="nsew", padx=(9, 0))
        run_host.columnconfigure(0, weight=1)
        run_host.rowconfigure(0, weight=1)
        run_canvas = tk.Canvas(run_host, highlightthickness=0)
        run_scrollbar = ttk.Scrollbar(
            run_host, orient="vertical", command=run_canvas.yview
        )
        run_panel = ttk.LabelFrame(
            run_canvas,
            text="Optimization run",
            style="Section.TLabelframe",
            padding=14,
        )
        run_window = run_canvas.create_window(
            (0, 0), window=run_panel, anchor="nw"
        )
        run_panel.bind(
            "<Configure>",
            lambda event: run_canvas.configure(scrollregion=run_canvas.bbox("all")),
        )
        run_canvas.bind(
            "<Configure>",
            lambda event: run_canvas.itemconfigure(run_window, width=event.width),
        )
        run_canvas.configure(yscrollcommand=run_scrollbar.set)
        run_canvas.grid(row=0, column=0, sticky="nsew")
        run_scrollbar.grid(row=0, column=1, sticky="ns")
        run_panel.columnconfigure(0, weight=1)

        ttk.Label(run_panel, text="Run inputs and active genes").grid(
            row=0, column=0, sticky="w"
        )
        self.optimization_fixed_var = tk.StringVar(value="Select this tab to load the current inputs.")
        ttk.Label(
            run_panel,
            textvariable=self.optimization_fixed_var,
            style="ResultValue.TLabel",
            justify="left",
            wraplength=420,
        ).grid(row=1, column=0, sticky="ew", pady=(5, 16))

        self.optimization_state_var = tk.StringVar(value="Ready to optimize")
        ttk.Label(run_panel, textvariable=self.optimization_state_var).grid(
            row=2, column=0, sticky="w"
        )
        self.optimization_progress = ttk.Progressbar(
            run_panel, mode="determinate", maximum=300, value=0
        )
        self.optimization_progress.grid(row=3, column=0, sticky="ew", pady=(7, 16))

        ttk.Label(run_panel, text="Optimization objective Cf").grid(
            row=4, column=0, sticky="w"
        )
        self.optimization_best_var = tk.StringVar(value="—")
        ttk.Label(
            run_panel, textvariable=self.optimization_best_var, style="ResultValue.TLabel"
        ).grid(row=5, column=0, sticky="w", pady=(3, 14))

        ttk.Label(run_panel, text="Best geometry in current generation").grid(
            row=6, column=0, sticky="w"
        )
        self.optimization_solution_var = tk.StringVar(value="—")
        ttk.Label(
            run_panel,
            textvariable=self.optimization_solution_var,
            style="ResultValue.TLabel",
            justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(3, 18))

        ttk.Label(
            run_panel,
            text=(
                "BLIMP-lite resolves Cebeci-Smith profiles. Quick uses a weak "
                "momentum-integral closure. MOC uses exact Kliegel-Levine MOC "
                "samples, a linear response surface, and exact refined finalists."
            ),
            wraplength=420,
        ).grid(row=8, column=0, sticky="ew", pady=(0, 8))

        buttons = ttk.Frame(run_panel)
        buttons.grid(row=9, column=0, sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)
        self.start_blimp_optimization_button = ttk.Button(
            buttons,
            text="Start with BLIMP-lite",
            style="Primary.TButton",
            command=lambda: self.optimization_action("blimp"),
        )
        self.start_blimp_optimization_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self.start_quick_optimization_button = ttk.Button(
            buttons,
            text="Start quick screening",
            style="Primary.TButton",
            command=lambda: self.optimization_action("quick"),
        )
        self.start_quick_optimization_button.grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.start_moc_optimization_button = ttk.Button(
            buttons,
            text="Start MOC-assisted optimization",
            style="Primary.TButton",
            command=lambda: self.optimization_action("moc"),
        )
        self.start_moc_optimization_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        self.cancel_optimization_button = ttk.Button(
            buttons,
            text="Cancel",
            command=self.cancel_optimization,
            state="disabled",
        )
        self.cancel_optimization_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )

    def _plot_optimization_history(self):
        """Draw fitness statistics and normalized best-gene trajectories."""
        fitness_ax, genes_ax = self.optimization_history_axes
        fitness_ax.clear()
        genes_ax.clear()
        generations = np.asarray(
            self.optimization_history["generation"], dtype=float
        )
        if generations.size == 0:
            fitness_ax.text(
                0.5, 0.5, "Fitness history appears during a run",
                ha="center", va="center", transform=fitness_ax.transAxes,
            )
            genes_ax.text(
                0.5, 0.5, "Best-gene trajectories appear here",
                ha="center", va="center", transform=genes_ax.transAxes,
            )
            fitness_ax.set_title("Population fitness")
            genes_ax.set_title("Best chromosome")
            genes_ax.set_xlabel("Generation")
            self.optimization_history_canvas.draw_idle()
            return

        best = np.asarray(self.optimization_history["best"], dtype=float)
        mean = np.asarray(self.optimization_history["mean"], dtype=float)
        std = np.asarray(self.optimization_history["std"], dtype=float)
        fitness_ax.plot(generations, best, color="#1f6f43", lw=2, label="Best")
        fitness_ax.plot(generations, mean, color="#315f9d", lw=1.6, label="Mean")
        fitness_ax.fill_between(
            generations, mean - std, mean + std, color="#315f9d", alpha=0.16,
            label="Mean ± std",
        )
        fitness_ax.set(ylabel="Effective Cf", title="Population fitness")
        fitness_ax.grid(alpha=0.25)
        fitness_ax.legend(loc="best", ncols=3)

        genes = np.asarray(self.optimization_history["genes"], dtype=float)
        try:
            limits = np.array([
                [float(self.optimization_entries["bell_fraction_min"].get()) / 100.0,
                 float(self.optimization_entries["bell_fraction_max"].get()) / 100.0],
                [float(self.optimization_entries["theta_in_min"].get()),
                 float(self.optimization_entries["theta_in_max"].get())],
            ])
        except ValueError:
            limits = np.array([
                [0.60, 1.00], [20.0, 45.0],
            ])
        normalized = 100.0 * (genes - limits[:, 0]) / np.maximum(
            limits[:, 1] - limits[:, 0], 1e-12
        )
        for index, label in enumerate(("bell_fraction", "theta_in")):
            genes_ax.plot(generations, normalized[:, index], lw=1.6, label=label)
        genes_ax.set(
            xlabel="Generation",
            ylabel="Position in search interval [%]",
            title="Best chromosome — normalized genes",
            ylim=(-5.0, 105.0),
        )
        genes_ax.grid(alpha=0.25)
        genes_ax.legend(loc="best", ncols=3)
        self.optimization_history_canvas.draw_idle()

    def show_optimization_tab(self):
        self.notebook.select(self.tab_frames["Optimization"])
        self._refresh_optimization_fixed_inputs()
        self.status.configure(text="Optimization settings ready")

    def _refresh_optimization_fixed_inputs(self):
        try:
            inputs = self.read_inputs()
            ideal_epsilon = calculate_ideal_expansion_ratio(inputs)
            self.optimization_ideal_epsilon_var.set(
                f"{ideal_epsilon:.6f}  Ae/At  (Pe = Pamb)"
            )
            self.optimization_fixed_var.set(
                f"Pc = {inputs.chamber_pressure_bar:.3g} bar    O/F = {inputs.mixture_ratio:.3g}\n"
                f"Pamb = {inputs.ambient_pressure_atm:.4g} atm    "
                f"Rt = {inputs.throat_radius_m * 1e3:.3g} mm\n"
                f"Chamber diameter = {inputs.chamber_diameter_m * 1e3:.3g} mm\n"
                f"CEA-sized epsilon = {ideal_epsilon:.6f} from Pe = Pamb\n"
                "Genes: bell fraction and theta_in\n"
                f"Final MOC grid: {self.optimization_moc_resolution.get()}\n"
                f"Fixed: theta_sub = {inputs.theta_sub_deg:.3g} deg    "
                f"α = {inputs.reference_half_angle_deg:.3g} deg"
            )
        except ValueError:
            self.optimization_ideal_epsilon_var.set("Unavailable for the current inputs")
            self.optimization_fixed_var.set("One or more inputs in the left panel are invalid.")

    def read_optimization_settings(
        self, boundary_layer_model: str = "blimp"
    ) -> OptimizationSettings:
        def number(key: str) -> float:
            return float(self.optimization_entries[key].get())

        def integer(key: str) -> int:
            raw = number(key)
            if not raw.is_integer():
                raise ValueError(f"{key.replace('_', ' ').title()} must be an integer.")
            return int(raw)

        try:
            moc_axial, moc_radial = MOC_OPTIMIZATION_RESOLUTIONS[
                self.optimization_moc_resolution.get()
            ]
        except KeyError as exc:
            raise ValueError("Select a valid final MOC resolution.") from exc

        settings = OptimizationSettings(
            bell_fraction_min=number("bell_fraction_min") / 100.0,
            bell_fraction_max=number("bell_fraction_max") / 100.0,
            theta_in_min_deg=number("theta_in_min"),
            theta_in_max_deg=number("theta_in_max"),
            num_generations=integer("num_generations"),
            population_size=integer("population_size"),
            num_parents_mating=integer("num_parents_mating"),
            keep_elitism=integer("keep_elitism"),
            saturation_generations=integer("saturation_generations"),
            crossover_probability=number("crossover_probability"),
            mutation_percent_high=integer("mutation_percent_high"),
            mutation_percent_low=integer("mutation_percent_low"),
            evaluation_mode=self.optimization_evaluation_mode.get().lower(),
            parallel_workers=integer("parallel_workers"),
            cache_evaluations=self.optimization_cache_evaluations.get(),
            boundary_layer_model=boundary_layer_model,
            moc_refine_axial_stations=moc_axial,
            moc_refine_radial_stations=moc_radial,
        )
        settings.validate()
        return settings

    def read_inputs(self) -> NozzleInputs:
        values = {key: float(entry.get()) for key, entry in self.entries.items()}
        return NozzleInputs(
            chamber_pressure_bar=values["chamber_pressure_bar"],
            mixture_ratio=values["mixture_ratio"],
            ambient_pressure_atm=values["ambient_pressure_atm"],
            expansion_ratio=values["expansion_ratio"],
            throat_radius_m=values["throat_radius_mm"] / 1000.0,
            chamber_diameter_m=values["chamber_diameter_mm"] / 1000.0,
            reference_half_angle_deg=values["reference_half_angle_deg"],
            bell_fraction=values["bell_fraction_percent"] / 100.0,
            theta_in_deg=values["theta_in_deg"],
            theta_sub_deg=values["theta_sub_deg"],
        )

    def run_simulation(self):
        try:
            self.status.configure(text="Generating geometry and updating engineering models…")
            self.update_idletasks()
            self.result = simulate(self.read_inputs())
            self._refresh_optimization_fixed_inputs()
            self._update_summary()
            self._update_plots()
            self.status.configure(text="Geometry generated successfully")
        except Exception as exc:  # noqa: BLE001 - GUI must present backend errors cleanly
            self.status.configure(text=f"Simulation failed: {exc}")
            messagebox.showerror("Simulation error", str(exc), parent=self)

    def read_moc_settings(self) -> MOCSettings:
        def integer(key: str) -> int:
            value = float(self.moc_entries[key].get())
            if not value.is_integer():
                raise ValueError(f"{key.replace('_', ' ').title()} must be an integer.")
            return int(value)

        settings = MOCSettings(
            axial_stations=integer("axial_stations"),
            radial_stations=integer("radial_stations"),
            initialization=(
                "kliegel_levine"
                if self.moc_initialization_var.get().startswith("Kliegel")
                else (
                    "sauer"
                    if self.moc_initialization_var.get().startswith("Sauer")
                    else "quasi_1d"
                )
            ),
            start_mach=float(self.moc_entries["start_mach"].get()),
        )
        settings.validate()
        return settings

    def run_moc_analysis(self):
        if self.moc_thread is not None and self.moc_thread.is_alive():
            return
        try:
            inputs = self.read_inputs()
            settings = self.read_moc_settings()
            if self.result is None or self.result.inputs != inputs:
                self.result = simulate(inputs)
                self._refresh_optimization_fixed_inputs()
                self._update_summary()
                self._update_plots()
        except Exception as exc:  # noqa: BLE001 - GUI reports backend validation errors
            messagebox.showerror("Invalid MOC inputs", str(exc), parent=self)
            return

        try:
            while True:
                self.moc_queue.get_nowait()
        except Empty:
            pass

        base_result = self.result
        self.moc_progress.configure(maximum=settings.axial_stations, value=1)
        self.moc_state_var.set(
            "Building the transonic initial line and characteristic transition net..."
            if settings.initialization in {"kliegel_levine", "sauer"}
            else f"Marching axisymmetric characteristics: 1/{settings.axial_stations} stations..."
        )
        self.status.configure(text="Running pressure-based axisymmetric MOC analysis...")
        self.generate_button.configure(state="disabled")
        self.moc_button.configure(state="disabled")
        self.moc_tab_button.configure(state="disabled")
        self.notebook.select(self.tab_frames["MOC analysis"])

        def report(current: int, total: int) -> None:
            self.moc_queue.put(("progress", current, total))

        def worker() -> None:
            try:
                moc_result = analyze_prescribed_nozzle(
                    base_result.inputs,
                    base_result.geometry,
                    base_result.cea,
                    friction_thrust_coefficient=(
                        base_result.performance.friction_thrust_coefficient
                    ),
                    settings=settings,
                    progress=report,
                )
                self.moc_queue.put(("complete", moc_result))
            except Exception as exc:  # noqa: BLE001 - transferred to the GUI thread
                self.moc_queue.put(("error", exc))

        self.moc_thread = Thread(target=worker, daemon=True)
        self.moc_thread.start()
        self.after(100, self._poll_moc)

    def _poll_moc(self):
        finished = False
        try:
            while True:
                message = self.moc_queue.get_nowait()
                if message[0] == "progress":
                    current, total = int(message[1]), int(message[2])
                    self.moc_progress.configure(maximum=total, value=current)
                    self.moc_state_var.set(
                        f"Transonic initialization and MOC march: {current}/{total} stations..."
                    )
                elif message[0] == "complete":
                    finished = True
                    moc_result = message[1]
                    self.result.moc = moc_result
                    self.moc_progress.configure(
                        maximum=moc_result.x_m.size,
                        value=moc_result.x_m.size,
                    )
                    verification = (
                        "verification target met"
                        if moc_result.mass_flow_residual <= 5.0e-3
                        else "preliminary: mass target not met"
                    )
                    self.moc_state_var.set(
                        f"MOC completed - mass residual {moc_result.mass_flow_residual:.3%}; "
                        f"Cf,MOC = {moc_result.inviscid_thrust_coefficient:.6f}; "
                        f"time = {moc_result.total_time_s:.2f} s ({verification})."
                    )
                    self._update_summary()
                    self._plot_moc()
                    self.canvases["MOC analysis"].draw_idle()
                    self.status.configure(
                        text=(
                            "Axisymmetric MOC completed - "
                            f"mass residual {moc_result.mass_flow_residual:.3%}"
                        )
                    )
                elif message[0] == "error":
                    finished = True
                    self.moc_state_var.set(f"MOC analysis failed: {message[1]}")
                    self.status.configure(text=f"MOC analysis failed: {message[1]}")
                    messagebox.showerror("MOC analysis error", str(message[1]), parent=self)
        except Empty:
            pass

        if finished:
            self.generate_button.configure(state="normal")
            self.moc_button.configure(state="normal")
            self.moc_tab_button.configure(state="normal")
        elif self.moc_thread is not None and self.moc_thread.is_alive():
            self.after(100, self._poll_moc)
        else:
            self.generate_button.configure(state="normal")
            self.moc_button.configure(state="normal")
            self.moc_tab_button.configure(state="normal")

    def optimization_action(self, boundary_layer_model: str = "blimp"):
        if self.optimization_thread is not None and self.optimization_thread.is_alive():
            return
        try:
            base_inputs = self.read_inputs()
            base_inputs.validate()
            settings = self.read_optimization_settings(boundary_layer_model)
        except ValueError as exc:
            messagebox.showerror("Invalid optimization inputs", str(exc), parent=self)
            return

        try:
            while True:
                self.optimization_queue.get_nowait()
        except Empty:
            pass
        self.optimization_cancel.clear()
        self.active_boundary_layer_model = boundary_layer_model
        for values in self.optimization_history.values():
            values.clear()
        self._plot_optimization_history()
        self.optimization_started_at = time.perf_counter()
        self.optimization_progress.configure(maximum=settings.num_generations, value=0)
        self.generate_button.configure(state="disabled")
        self._set_optimization_controls_running(True)
        self._refresh_optimization_fixed_inputs()
        model_label = {
            "blimp": "BLIMP-lite",
            "quick": "Quick",
            "moc": "MOC-assisted",
        }[boundary_layer_model]
        self.optimization_state_var.set(
            f"Starting {model_label} generation 0/{settings.num_generations}…"
        )
        self.optimization_best_var.set("—")
        self.optimization_solution_var.set("—")
        self.status.configure(
            text=f"Optimizing bell fraction and initial angle with {model_label}…"
        )

        def report(
            generation,
            total,
            fitness,
            mean_fitness,
            std_fitness,
            expansion_ratio,
            bell_fraction,
            theta_in,
            theta_out,
            exit_pressure_bar,
        ):
            self.optimization_queue.put(
                (
                    "progress",
                    generation,
                    total,
                    fitness,
                    mean_fitness,
                    std_fitness,
                    expansion_ratio,
                    bell_fraction,
                    theta_in,
                    theta_out,
                    exit_pressure_bar,
                )
            )

        def worker():
            try:
                result = optimize_geometry(
                    base_inputs,
                    settings=settings,
                    progress=report,
                    cancel_event=self.optimization_cancel,
                    status=lambda message: self.optimization_queue.put(
                        ("status", message)
                    ),
                )
                self.optimization_queue.put(("complete", result))
            except Exception as exc:  # noqa: BLE001 - transferred to the GUI thread
                self.optimization_queue.put(("error", exc))

        self.optimization_thread = Thread(target=worker, daemon=True)
        self.optimization_thread.start()
        self.after(100, self._poll_optimization)

    def cancel_optimization(self):
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            return
        self.optimization_cancel.set()
        self.cancel_optimization_button.configure(state="disabled", text="Stopping…")
        self.optimization_state_var.set("Stopping after the current generation…")
        self.status.configure(text="Stopping optimization after the current generation…")

    def _set_optimization_controls_running(self, running: bool):
        entry_state = "disabled" if running else "normal"
        for entry in self.optimization_entries.values():
            entry.configure(state=entry_state)
        button_state = "disabled" if running else "normal"
        self.start_blimp_optimization_button.configure(state=button_state)
        self.start_quick_optimization_button.configure(state=button_state)
        self.start_moc_optimization_button.configure(state=button_state)
        self.optimization_mode_combo.configure(state="disabled" if running else "readonly")
        self.optimization_moc_resolution_combo.configure(
            state="disabled" if running else "readonly"
        )
        self.optimization_cache_check.configure(state="disabled" if running else "normal")
        self.cancel_optimization_button.configure(
            state="normal" if running else "disabled", text="Cancel"
        )

    def _poll_optimization(self):
        finished = False
        try:
            while True:
                message = self.optimization_queue.get_nowait()
                if message[0] == "progress":
                    (
                        _,
                        generation,
                        total,
                        fitness,
                        mean_fitness,
                        std_fitness,
                        expansion_ratio,
                        bell_fraction,
                        theta_in,
                        theta_out,
                        exit_pressure_bar,
                    ) = message
                    self.optimization_progress.configure(maximum=total, value=generation)
                    elapsed = (
                        time.perf_counter() - self.optimization_started_at
                        if self.optimization_started_at is not None
                        else 0.0
                    )
                    rate = generation / elapsed if elapsed > 0.0 else 0.0
                    eta = (total - generation) / rate if rate > 0.0 else 0.0
                    percentage = generation / total * 100.0
                    self.optimization_history["generation"].append(generation)
                    self.optimization_history["best"].append(fitness)
                    self.optimization_history["mean"].append(mean_fitness)
                    self.optimization_history["std"].append(std_fitness)
                    self.optimization_history["genes"].append(
                        [bell_fraction, theta_in]
                    )
                    # Keep the main input panel synchronized with the best
                    # chromosome.  A full BLIMP regeneration is intentionally
                    # deferred until completion/cancellation to keep the GUI
                    # responsive while the optimizer is running.
                    self._write_optimized_input_values(
                        expansion_ratio, bell_fraction, theta_in
                    )
                    self._plot_optimization_history()
                    self.optimization_state_var.set(
                        f"{generation}/{total}  {percentage:5.1f}%  |  "
                        f"elapsed {self._format_duration(elapsed)}  |  "
                        f"ETA {self._format_duration(eta)}  |  {rate:.2f} gen/s"
                    )
                    self.optimization_best_var.set(f"{fitness:.8f}")
                    self.optimization_solution_var.set(
                        f"Expansion     {expansion_ratio:9.4f}\n"
                        f"Exit pressure {exit_pressure_bar:9.4f} bar\n"
                        f"Bell fraction {bell_fraction * 100.0:9.4f} %\n"
                        f"Initial angle {theta_in:9.4f} deg\n"
                        f"Derived exit {theta_out:9.4f} deg"
                    )
                    self.status.configure(
                        text=f"Optimizing geometry ({self.active_boundary_layer_model}) — "
                        f"generation {generation}/{total}, best merit {fitness:.6f}"
                    )
                elif message[0] == "status":
                    self.optimization_state_var.set(str(message[1]))
                    self.status.configure(text=str(message[1]))
                elif message[0] == "complete":
                    finished = True
                    self._apply_optimized_geometry(
                        message[1], cancelled=self.optimization_cancel.is_set()
                    )
                elif message[0] == "error":
                    finished = True
                    self.optimization_state_var.set(f"Failed: {message[1]}")
                    self.status.configure(text=f"Optimization failed: {message[1]}")
                    messagebox.showerror("Optimization error", str(message[1]), parent=self)
        except Empty:
            pass

        if finished:
            self.generate_button.configure(state="normal")
            self._set_optimization_controls_running(False)
            self.optimization_thread = None
        elif self.optimization_thread is not None and self.optimization_thread.is_alive():
            self.after(100, self._poll_optimization)
        else:
            self.generate_button.configure(state="normal")
            self._set_optimization_controls_running(False)
            self.optimization_thread = None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_optimized_input_values(
        self,
        expansion_ratio: float,
        bell_fraction: float,
        theta_in_deg: float,
    ):
        """Write a canonical, non-redundant optimized contour to the main inputs."""
        optimized_values = {
            "expansion_ratio": expansion_ratio,
            "bell_fraction_percent": bell_fraction * 100.0,
            "theta_in_deg": theta_in_deg,
        }
        for key, value in optimized_values.items():
            entry = self.entries[key]
            entry.delete(0, "end")
            entry.insert(0, f"{value:.6f}")

    def _apply_optimized_geometry(
        self, result: OptimizationResult, cancelled: bool = False
    ):
        self.optimization_result = result
        self._write_optimized_input_values(
            result.expansion_ratio,
            result.bell_fraction,
            result.theta_in_deg,
        )
        self.run_simulation()
        model_label = {
            "blimp": "BLIMP-lite",
            "quick": "Quick",
            "moc": "MOC-assisted",
        }[result.boundary_layer_model]
        if result.moc_result is not None:
            self.result.moc = result.moc_result
            self._update_summary()
            self._update_plots()
            validated_fitness = result.moc_result.friction_corrected_thrust_coefficient
            validation_label = "refined MOC"
        else:
            validated_fitness = self.result.performance.effective_thrust_coefficient
            validation_label = "BLIMP-lite"
        outcome = "Stopped" if cancelled else "Completed"
        self.optimization_state_var.set(
            f"{outcome} {model_label} search after {result.generations_completed} generations; "
            f"best geometry evaluated with {validation_label}"
        )
        self.optimization_best_var.set(
            f"Search ({model_label})  {result.fitness:.8f}\n"
            f"Final {validation_label:<12} {validated_fitness:.8f}"
        )
        self.optimization_solution_var.set(
            f"Expansion     {result.expansion_ratio:9.4f}\n"
            f"Exit pressure {result.exit_pressure_bar:9.4f} bar\n"
            f"Bell fraction {result.bell_fraction * 100.0:9.4f} %\n"
            f"Initial angle {result.theta_in_deg:9.4f} deg\n"
            f"Fixed conv.   {result.theta_sub_deg:9.4f} deg\n"
            f"Derived exit {result.theta_out_deg:9.4f} deg"
        )
        if result.moc_result is not None:
            residual = result.moc_mass_flow_residual or 0.0
            self.optimization_solution_var.set(
                self.optimization_solution_var.get()
                + f"\nMOC residual  {residual:9.4%}"
                + f"\nMOC grid      {result.moc_result.x_m.size:4d} x "
                f"{result.moc_result.radial_fraction.size:d}"
                + f"\nMOC DOE       {result.moc_training_evaluations:9d} runs"
            )
            self.notebook.select(self.tab_frames["MOC analysis"])
        else:
            self.notebook.select(self.tab_frames["Geometry 2D"])
        self.status.configure(
            text=f"{model_label} optimization {outcome.lower()} after "
            f"{result.generations_completed} generations - final {validation_label} Cf "
            f"{validated_fitness:.6f}"
        )

    def _update_summary(self):
        r = self.result
        g, c, bl, performance = r.geometry, r.cea, r.boundary_layer, r.performance
        ideal_eps = (
            f"{c.ideal_expansion_ratio:9.4f}"
            if c.ideal_expansion_ratio is not None
            else "      n/a"
        )
        text = (
            f"CEA PROPERTIES\n"
            f"Tc             {c.chamber.temperature_k:9.2f} K\n"
            f"Tt             {c.throat.temperature_k:9.2f} K\n"
            f"Te             {c.exit.temperature_k:9.2f} K\n"
            f"gamma c/t/e    {c.chamber.gamma:.4f} / {c.throat.gamma:.4f} / {c.exit.gamma:.4f}\n"
            f"MW c/t/e       {c.chamber.molecular_weight_g_mol:.3f} / "
            f"{c.throat.molecular_weight_g_mol:.3f} / {c.exit.molecular_weight_g_mol:.3f} g/mol\n"
            f"c*             {c.cstar_m_s:9.2f} m/s\n"
            f"Exit Mach      {c.exit_mach:9.4f}\n"
            f"Exit pressure  {c.exit_pressure_bar:9.4f} bar\n\n"
            f"AMBIENT PERFORMANCE\n"
            f"Pamb           {r.inputs.ambient_pressure_atm:9.4f} atm\n"
            f"Ideal epsilon  {ideal_eps}\n"
            f"CEA Cf mom.    {c.ideal_momentum_thrust_coefficient:9.6f}\n"
            f"CEA Cf ambient {c.ambient_thrust_coefficient:9.6f}\n"
            f"Flow mode      {c.ambient_mode}\n"
            f"Eta divergence {performance.divergence_efficiency:9.6f}\n"
            f"Eta momentum   {performance.momentum_efficiency:9.6f}\n"
            f"Cf momentum    {performance.momentum_thrust_coefficient:9.6f}\n"
            f"Cf pressure    {performance.pressure_thrust_coefficient:9.6f}\n"
            f"Cf friction    {performance.friction_thrust_coefficient:9.6f}\n"
            f"Cf effective   {performance.effective_thrust_coefficient:9.6f}\n"
            f"Thrust eff.    {performance.effective_thrust_n:9.2f} N\n\n"
            f"GEOMETRY\n"
            f"Exit radius    {g.exit_radius_m*1e3:9.3f} mm\n"
            f"Divergent L    {g.divergent_length_m*1e3:9.3f} mm\n"
            f"Total L        {g.total_length_m*1e3:9.3f} mm\n"
            f"Convergent ang {r.inputs.theta_sub_deg:9.3f} deg\n"
            f"Exit angle     {g.theta_out_deg:9.3f} deg\n"
            f"Contraction    {g.contraction_ratio:9.3f}\n"
            f"BL eta_v       {bl.velocity_efficiency:9.5f}\n"
            f"Exit delta*    {bl.displacement_thickness_m[-1]*1e3:9.4f} mm\n"
            f"Adiabatic Tw,e {r.thermal.wall_temperature_k[-1]:9.2f} K\n"
            f"Wall condition adiabatic (Tw = Tr, qw = 0)\n"
        )
        if r.moc is not None:
            moc = r.moc
            text += (
                f"\nAXISYMMETRIC MOC\n"
                f"Initialization {moc.initialization}\n"
                f"Grid           {moc.x_m.size:d} x {moc.radial_fraction.size:d}\n"
                f"gamma fixed    {moc.gamma:9.5f}\n"
                f"March M min    {moc.start_mach:9.5f}\n"
                f"Data-line Mmin {np.min(moc.initial_line_mach):9.5f}\n"
                f"Sauer eta      {moc.sauer_eta_m*1e3:9.4f} mm\n"
                f"CEA choked mdot {moc.cea_choked_mass_flow_kg_s:8.5f} kg/s\n"
                f"Initial mdot err {moc.initial_mass_flow_error:8.4%}\n"
                f"Mass residual  {moc.mass_flow_residual:9.4%}\n"
                f"Cf MOC invisc. {moc.inviscid_thrust_coefficient:9.6f}\n"
                f"Cf MOC - fric. {moc.friction_corrected_thrust_coefficient:9.6f}\n"
                f"Exit M range   {np.min(moc.mach[-1]):.4f} / {np.max(moc.mach[-1]):.4f}\n"
                f"Exit Pe range  {np.min(moc.exit_pressure_bar):.4f} / "
                f"{np.max(moc.exit_pressure_bar):.4f} bar\n"
                f"Exit theta max {np.max(np.abs(moc.exit_theta_deg)):9.4f} deg\n"
                f"Init / march   {moc.initialization_time_s:7.2f} / "
                f"{moc.marching_time_s:7.2f} s\n"
                f"Total time     {moc.total_time_s:9.2f} s\n"
            )
            if moc.warnings:
                text += "MOC warning     " + " | ".join(moc.warnings) + "\n"
        if self.optimization_result is not None:
            opt = self.optimization_result
            final_label = "Final MOC Cf  " if opt.moc_result is not None else "Final BLIMP Cf"
            final_cf = (
                opt.moc_result.friction_corrected_thrust_coefficient
                if opt.moc_result is not None
                else performance.effective_thrust_coefficient
            )
            text += (
                f"\nOPTIMIZATION\n"
                f"Search model   {opt.boundary_layer_model}\n"
                f"Search Cf      {opt.fitness:9.6f}\n"
                f"{final_label} {final_cf:9.6f}\n"
                f"Expansion      {opt.expansion_ratio:9.4f}\n"
                f"Bell fraction  {opt.bell_fraction:9.4f}\n"
                f"Theta in       {opt.theta_in_deg:9.4f} deg\n"
                f"Theta sub fixed{opt.theta_sub_deg:9.4f} deg\n"
                f"Pe             {opt.exit_pressure_bar:9.4f} bar\n"
                f"Generations    {opt.generations_completed:9d}\n"
            )
            if opt.moc_mass_flow_residual is not None:
                text += (
                    f"MOC mass resid {opt.moc_mass_flow_residual:9.4%}\n"
                    f"MOC DOE runs   {opt.moc_training_evaluations:9d}\n"
                )
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    def _update_plots(self):
        self._plot_expansion_sizing()
        self._plot_geometry()
        self._plot_3d()
        self._plot_moc()
        self._plot_flow()
        self._plot_thermal()
        self._plot_boundary_layer()
        self._plot_loss_breakdown()
        for canvas in self.canvases.values():
            canvas.draw_idle()

    def _plot_expansion_sizing(self):
        fig = self.figures["Expansion sizing"]
        fig.clear()
        ax = fig.add_subplot(111)
        current_eps = self.result.inputs.expansion_ratio
        ideal_eps = self.result.cea.ideal_expansion_ratio
        reference_eps = ideal_eps if ideal_eps is not None else current_eps
        eps_min = max(1.05, 0.55 * min(current_eps, reference_eps))
        eps_max = max(1.25 * eps_min, 1.65 * max(current_eps, reference_eps))
        expansion_ratios = np.linspace(eps_min, eps_max, 90)
        exit_pressure_bar = np.asarray(
            calculate_exit_pressure_curve(
                self.result.inputs.chamber_pressure_bar,
                self.result.inputs.mixture_ratio,
                expansion_ratios,
            )
        )
        exit_pressure_atm = exit_pressure_bar / ATM_TO_BAR
        pamb_atm = self.result.inputs.ambient_pressure_atm
        ax.plot(expansion_ratios, exit_pressure_atm, color="#245a9b", lw=2, label="Pe(ε)")
        ax.axhline(pamb_atm, color="#b33a3a", ls="--", label=f"Pamb = {pamb_atm:.4g} atm")
        if ideal_eps is not None and eps_min <= ideal_eps <= eps_max:
            ax.scatter([ideal_eps], [pamb_atm], color="#b33a3a", zorder=4)
            ax.axvline(ideal_eps, color="#777", ls=":", label=f"Ideal ε = {ideal_eps:.4f}")
        ax.scatter(
            [current_eps],
            [self.result.cea.exit_pressure_bar / ATM_TO_BAR],
            color="#24824b",
            zorder=4,
            label=f"Current ε = {current_eps:.4f}",
        )
        ax.set(
            title="CEA exit pressure versus expansion ratio",
            xlabel="Expansion ratio ε = Ae/At [-]",
            ylabel="Pressure [atm]",
            xlim=(eps_min, eps_max),
        )
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    def _plot_geometry(self):
        fig = self.figures["Geometry 2D"]; fig.clear(); ax = fig.add_subplot(111)
        colors = ("#b33a3a", "#3875b5", "#2e8b57", "#202020")
        for (label, (x, radius)), color in zip(self.result.geometry.segments.items(), colors):
            ax.plot(x * 1e3, radius * 1e3, color=color, lw=2, label=label)
            ax.plot(x * 1e3, -radius * 1e3, color=color, lw=2)
        ax.axvline(self.result.geometry.throat_x_m * 1e3, color="#888", ls="--", lw=1)
        ax.set(title="Axisymmetric bell-nozzle contour", xlabel="Axial position [mm]", ylabel="Radius [mm]")
        exit_x_mm = self.result.geometry.exit_x_m * 1e3
        max_radius_mm = np.max(self.result.geometry.radius_m) * 1e3
        ax.set_xlim(0.0, exit_x_mm * 1.03)
        ax.set_ylim(-max_radius_mm * 1.08, max_radius_mm * 1.08)
        ax.set_aspect("auto")
        ax.margins(x=0.0, y=0.0)
        ax.grid(alpha=0.25); ax.legend(loc="best")

    def _plot_3d(self):
        fig = self.figures["Interactive 3D"]; fig.clear(); ax = fig.add_subplot(111, projection="3d")
        x = self.result.geometry.x_m[::2]
        radius = self.result.geometry.radius_m[::2]
        theta = np.linspace(0, 2 * np.pi, 72)
        x_mesh, theta_mesh = np.meshgrid(x, theta, indexing="ij")
        r_mesh = np.tile(radius, (theta.size, 1)).T
        ax.plot_surface(x_mesh * 1e3, r_mesh * np.cos(theta_mesh) * 1e3,
                        r_mesh * np.sin(theta_mesh) * 1e3, cmap="plasma", linewidth=0)
        ax.set(title="Revolved nozzle surface", xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
        ax.set_box_aspect((3.2, 1, 1)); ax.view_init(elev=20, azim=40)

    def _plot_moc(self):
        if "MOC analysis" not in self.figures:
            return
        fig = self.figures["MOC analysis"]
        fig.clear()
        if self.result is None or self.result.moc is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.55,
                "Run the axisymmetric MOC analysis to resolve M(x,r), p(x,r) and exit angularity.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.text(
                0.5,
                0.45,
                "The fixed-x solver cannot yet march directly from a curved transonic line. "
                "Kliegel-Levine is marched through a curved characteristic transition net "
                "before the fixed-x field. Quasi-1D remains a reference and projected "
                "Sauer is diagnostic only. Always check the reported mass-flow residual "
                "before using Cf for design.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return

        moc = self.result.moc
        axes = fig.subplots(2, 2)
        x_grid = np.broadcast_to(moc.x_m[:, None], moc.radius_m.shape) * 1.0e3
        radius_mm = moc.radius_m * 1.0e3

        mach_map = axes[0, 0].pcolormesh(
            x_grid, radius_mm, moc.mach, shading="auto", cmap="viridis"
        )
        fig.colorbar(mach_map, ax=axes[0, 0], label="Mach [-]")
        axes[0, 0].plot(moc.x_m * 1.0e3, moc.radius_m[:, -1] * 1.0e3, "k", lw=1.2)
        if moc.initial_line_x_m.size:
            axes[0, 0].plot(
                moc.initial_line_x_m * 1.0e3,
                moc.initial_line_radius_m * 1.0e3,
                color="white",
                ls="--",
                lw=2.0,
                label="Initial-data line",
            )
            axes[0, 0].legend(loc="best")
        for line_x, line_radius in zip(
            moc.transition_line_x_m, moc.transition_line_radius_m
        ):
            axes[0, 0].plot(
                line_x * 1.0e3,
                line_radius * 1.0e3,
                color="white",
                lw=0.65,
                alpha=0.72,
            )
        axes[0, 0].set(title="Axisymmetric Mach field", ylabel="Radius [mm]")

        pressure_map = axes[0, 1].pcolormesh(
            x_grid,
            radius_mm,
            moc.pressure_pa / 1.0e5,
            shading="auto",
            cmap="plasma",
        )
        fig.colorbar(pressure_map, ax=axes[0, 1], label="Pressure [bar]")
        axes[0, 1].plot(moc.x_m * 1.0e3, moc.radius_m[:, -1] * 1.0e3, "k", lw=1.2)
        if moc.initial_line_x_m.size:
            axes[0, 1].plot(
                moc.initial_line_x_m * 1.0e3,
                moc.initial_line_radius_m * 1.0e3,
                color="white",
                ls="--",
                lw=2.0,
            )
        for line_x, line_radius in zip(
            moc.transition_line_x_m, moc.transition_line_radius_m
        ):
            axes[0, 1].plot(
                line_x * 1.0e3,
                line_radius * 1.0e3,
                color="white",
                lw=0.65,
                alpha=0.72,
            )
        axes[0, 1].set(title="Static-pressure field")

        exit_radius_fraction = moc.radial_fraction
        axes[1, 0].plot(
            moc.mach[-1], exit_radius_fraction, color="#245a9b", lw=2, label="MOC Mach"
        )
        axes[1, 0].axvline(
            self.result.flow.mach[-1], color="#245a9b", ls="--", label="Quasi-1D Mach"
        )
        axes[1, 0].set(
            title="Exit radial profiles",
            xlabel="Mach [-]",
            ylabel="r/Re [-]",
        )
        theta_axis = axes[1, 0].twiny()
        theta_axis.plot(
            moc.exit_theta_deg,
            exit_radius_fraction,
            color="#b33a3a",
            lw=1.8,
            label="MOC theta",
        )
        theta_axis.set_xlabel("Flow angle [deg]", color="#b33a3a")
        lines_1, labels_1 = axes[1, 0].get_legend_handles_labels()
        lines_2, labels_2 = theta_axis.get_legend_handles_labels()
        axes[1, 0].legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        normalized_mass_flow = moc.mass_flow_kg_s / moc.mass_flow_kg_s[0]
        axes[1, 1].plot(
            moc.x_m * 1.0e3,
            normalized_mass_flow,
            color="#2e6849",
            lw=2,
        )
        axes[1, 1].axhline(1.0, color="#777", ls="--", lw=1)
        axes[1, 1].set(
            title=(
                f"Mass conservation - residual {moc.mass_flow_residual:.3%}; "
                f"initial/CEA error {moc.initial_mass_flow_error:.3%}\n"
                f"Cf,MOC={moc.inviscid_thrust_coefficient:.6f}; "
                f"Cf,MOC-friction={moc.friction_corrected_thrust_coefficient:.6f}; "
                f"current Cf,eff={self.result.performance.effective_thrust_coefficient:.6f}\n"
                f"initialization={moc.initialization_time_s:.2f} s; "
                f"march={moc.marching_time_s:.2f} s"
            ),
            xlabel="Axial position [mm]",
            ylabel="mdot / mdot(start) [-]",
        )
        for ax in axes.flat:
            ax.grid(alpha=0.22)

    def _plot_flow(self):
        fig = self.figures["Flow profiles"]; fig.clear()
        x = self.result.geometry.x_m * 1e3; flow = self.result.flow
        axes = fig.subplots(3, 1, sharex=True)
        for ax, values, ylabel, color in zip(
            axes,
            (flow.mach, flow.temperature_k, flow.pressure_bar),
            ("Mach [-]", "Temperature [K]", "Pressure [bar]"),
            ("#244a9b", "#d05a33", "#6a3d9a"),
        ):
            ax.plot(x, values, color=color, lw=2); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
            ax.axvline(self.result.geometry.throat_x_m * 1e3, color="#888", ls="--", lw=1)
        axes[-1].set_xlabel("Axial position [mm]")

    def _plot_thermal(self):
        fig = self.figures["Thermal"]; fig.clear(); x = self.result.geometry.x_m * 1e3
        axes = fig.subplots(2, 1, sharex=True)
        axes[0].plot(x, self.result.thermal.wall_temperature_k, color="#b33a3a", label="Tw = Tr")
        axes[0].plot(x, self.result.flow.temperature_k, color="#315f9d", label="Te")
        axes[0].set_ylabel("Temperature [K]"); axes[0].grid(alpha=0.25); axes[0].legend()
        axes[1].plot(x, self.result.thermal.heat_transfer_coefficient_w_m2_k, color="#007c83")
        axes[1].set(ylabel="Bartz hg [W/m²K]", xlabel="Axial position [mm]"); axes[1].grid(alpha=0.25)

    def _plot_boundary_layer(self):
        fig = self.figures["Boundary layer"]; fig.clear(); x = self.result.geometry.x_m * 1e3
        axes = fig.subplots(2, 1, sharex=True); bl = self.result.boundary_layer
        axes[0].plot(
            x, bl.displacement_thickness_m * 1e3, color="#c27616", label="δ*"
        )
        axes[0].plot(
            x, bl.momentum_thickness_m * 1e3, color="#315f9d", label="θ"
        )
        axes[0].set_ylabel("Integral thickness [mm]")
        axes[0].grid(alpha=0.25); axes[0].legend(loc="best")
        axes[1].semilogy(x, bl.reynolds, color="#2e6849", label="Re_s")
        axes[1].semilogy(
            x, bl.skin_friction_coefficient, color="#8f3d72", label="Cf"
        )
        axes[1].set(ylabel="Re / Cf [-]", xlabel="Axial position [mm]")
        axes[1].grid(alpha=0.25); axes[1].legend(loc="best")

    def _plot_loss_breakdown(self):
        fig = self.figures["Loss breakdown"]
        fig.clear()
        ax = fig.add_subplot(111)
        contributions = loss_breakdown(self.result.performance)
        labels = list(contributions)
        values = np.array(list(contributions.values()), dtype=float)
        total = float(np.sum(values))
        percentages = 100.0 * values / total if total > 0.0 else np.zeros_like(values)
        colors = ("#d05a33", "#c8942f", "#8f3d72", "#315f9d")
        positions = np.arange(len(labels))
        bars = ax.barh(positions, percentages, color=colors, height=0.62)
        ax.set_yticks(positions, labels=labels)
        ax.invert_yaxis()
        right_limit = max(105.0, float(np.max(percentages)) * 1.28 if total else 105.0)
        ax.set_xlim(0.0, right_limit)
        for bar, percentage, value in zip(bars, percentages, values):
            ax.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{percentage:5.1f}%   (ΔCf = {value:.6f})",
                va="center",
                fontfamily="monospace",
            )
        ax.set(
            title=f"Modeled loss allocation — total ΔCf = {total:.6f}",
            xlabel="Share of total modeled losses [%]",
        )
        pressure_cf = self.result.performance.pressure_thrust_coefficient
        note = (
            "Shares use non-negative losses. "
            "Divergence and BL displacement are attributed sequentially; wall friction "
            "comes from the integrated shear stress."
        )
        if pressure_cf > 0.0:
            note += f" The ambient pressure term is a +{pressure_cf:.6f} gain and is excluded."
        ax.text(0.0, -0.13, note, transform=ax.transAxes, va="top", wrap=True)
        ax.grid(axis="x", alpha=0.25)

    def export_results(self):
        if self.result is None:
            messagebox.showinfo("No results", "Generate a geometry before exporting.", parent=self)
            return
        initial = Path.cwd() / "outputs"
        initial.mkdir(exist_ok=True)
        destination = filedialog.askdirectory(
            title="Choose the parent output folder", initialdir=initial, parent=self
        )
        if destination:
            output = export_result(self.result, destination)
            self.status.configure(text=f"Results exported to {output}")
            messagebox.showinfo("Export complete", str(output), parent=self)

    def reset_defaults(self):
        defaults = {key: default for fields in FIELD_GROUPS.values() for key, _, _, default in fields}
        for key, entry in self.entries.items():
            entry.delete(0, "end"); entry.insert(0, defaults[key])
        optimization_defaults = {}
        for key, _, _, lower, upper in OPTIMIZATION_RANGE_FIELDS:
            optimization_defaults[f"{key}_min"] = lower
            optimization_defaults[f"{key}_max"] = upper
        optimization_defaults.update(
            {key: default for key, _, default in OPTIMIZATION_GA_FIELDS}
        )
        optimization_defaults["parallel_workers"] = "4"
        for key, entry in self.optimization_entries.items():
            entry.delete(0, "end")
            entry.insert(0, optimization_defaults[key])
        moc_defaults = {
            "axial_stations": "360",
            "radial_stations": "41",
            "start_mach": "1.12",
        }
        self.moc_initialization_var.set("Kliegel-Levine characteristic net")
        self.optimization_moc_resolution.set(DEFAULT_MOC_OPTIMIZATION_RESOLUTION)
        for key, entry in self.moc_entries.items():
            entry.delete(0, "end")
            entry.insert(0, moc_defaults[key])
        self.moc_progress.configure(maximum=360, value=0)
        self.moc_state_var.set(
            "Generate the geometry, then run the axisymmetric analysis."
        )
        self.optimization_result = None
        self.active_boundary_layer_model = "blimp"
        self.optimization_progress.configure(maximum=300, value=0)
        self.optimization_state_var.set("Ready to optimize")
        self.optimization_best_var.set("—")
        self.optimization_solution_var.set("—")
        self.optimization_evaluation_mode.set("Processes")
        self.optimization_cache_evaluations.set(True)
        for values in self.optimization_history.values():
            values.clear()
        self._plot_optimization_history()
        self.run_simulation()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    NozzleSimulatorApp().mainloop()


if __name__ == "__main__":
    main()
