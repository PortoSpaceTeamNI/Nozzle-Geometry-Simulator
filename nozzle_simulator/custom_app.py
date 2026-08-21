"""Conservative visual test for the established desktop interface.

The complete layout, tab order, controls, plots, callbacks and scientific
workflow come directly from :class:`nozzle_simulator.app.NozzleSimulatorApp`.
This module changes only the widget theme. It deliberately does not introduce
new pages, dashboards or alternative engineering plots.
"""

from __future__ import annotations

from tkinter import ttk

import customtkinter as ctk

from .app import NozzleSimulatorApp


class NozzleSimulatorCustomApp(NozzleSimulatorApp):
    """The production window with a restrained, CustomTkinter-informed theme."""

    def __init__(self) -> None:
        # Retaining the original ttk widget tree guarantees the exact same
        # windows and behaviour as the established application.
        ctk.set_appearance_mode("Light")
        super().__init__()
        self.title("Rocket Nozzle Simulator - visual style test")
        self.configure(background="#E6E4DF")
        self.summary.configure(
            background="#FAF9F6",
            foreground="#24272A",
            selectbackground="#B9C5CD",
            selectforeground="#1D2023",
            insertbackground="#24272A",
            relief="solid",
            borderwidth=1,
        )

    def _configure_style(self) -> None:
        """Apply a sober engineering palette without changing the UI tree."""
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        background = "#E6E4DF"
        surface = "#F4F2ED"
        surface_high = "#FAF9F6"
        border = "#AAA8A2"
        text = "#24272A"
        muted = "#60656A"
        accent = "#526A7C"
        accent_active = "#405665"
        selection = "#D4DBDF"

        self.option_add("*Font", ("Segoe UI", 9))
        self.option_add("*Background", background)
        self.option_add("*Foreground", text)
        self.option_add("*insertBackground", text)
        self.option_add("*selectBackground", selection)
        self.option_add("*selectForeground", text)

        style.configure(".", background=background, foreground=text)
        style.configure("TFrame", background=background)
        style.configure("TLabel", background=background, foreground=text)
        style.configure(
            "Title.TLabel",
            background=background,
            foreground="#202326",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=background,
            foreground="#202326",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "ResultValue.TLabel",
            background=background,
            foreground="#334A5B",
            font=("Consolas", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#DAD8D3",
            foreground=muted,
            padding=(9, 5),
        )

        style.configure(
            "TLabelframe",
            background=surface,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=surface,
            foreground=text,
            font=("Segoe UI", 9, "bold"),
            padding=(3, 1),
        )
        style.configure(
            "Section.TLabelframe",
            background=surface,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=surface,
            foreground=text,
            font=("Segoe UI", 10, "bold"),
        )

        style.configure(
            "TEntry",
            fieldbackground=surface_high,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            insertcolor=text,
            padding=(6, 5),
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", accent)],
            lightcolor=[("focus", accent)],
            darkcolor=[("focus", accent)],
        )

        style.configure(
            "TButton",
            background=surface,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            relief="solid",
            padding=(10, 7),
        )
        style.map(
            "TButton",
            background=[("active", selection), ("disabled", "#D8D6D1")],
            foreground=[("disabled", "#8A8D90")],
            bordercolor=[("focus", accent)],
        )
        style.configure(
            "Primary.TButton",
            background=accent,
            foreground="#FAF9F6",
            bordercolor=accent_active,
            lightcolor=accent,
            darkcolor=accent,
            relief="solid",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", accent_active), ("disabled", "#AEB5BA")],
            foreground=[("disabled", "#ECEAE5")],
        )

        style.configure(
            "TNotebook",
            background=background,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            tabmargins=(1, 3, 1, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background="#D8D6D1",
            foreground="#41464A",
            bordercolor=border,
            padding=(11, 7),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", surface_high), ("active", selection)],
            foreground=[("selected", "#202326")],
            expand=[("selected", (0, 0, 0, 1))],
        )

        style.configure(
            "TCombobox",
            fieldbackground=surface_high,
            background=surface,
            foreground=text,
            arrowcolor=muted,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            padding=(5, 4),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", surface_high)],
            selectbackground=[("readonly", surface_high)],
            selectforeground=[("readonly", text)],
            bordercolor=[("focus", accent)],
        )
        style.configure(
            "TCheckbutton",
            background=background,
            foreground=text,
            indicatorcolor=surface_high,
            bordercolor=border,
            padding=(2, 3),
        )
        style.map(
            "TCheckbutton",
            indicatorcolor=[("selected", accent), ("active", selection)],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=accent,
            troughcolor="#D0CEC9",
            bordercolor=border,
            lightcolor=accent,
            darkcolor=accent,
        )
        style.configure(
            "Vertical.TScrollbar",
            background="#C7C5C0",
            troughcolor=background,
            bordercolor=background,
            arrowcolor=muted,
        )
        style.configure(
            "Horizontal.TScrollbar",
            background="#C7C5C0",
            troughcolor=background,
            bordercolor=background,
            arrowcolor=muted,
        )


def main() -> None:
    NozzleSimulatorCustomApp().mainloop()


if __name__ == "__main__":
    main()
