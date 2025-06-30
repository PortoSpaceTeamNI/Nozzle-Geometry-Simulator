[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

# Rocket Nozzle Geometry Simulator

This simulator was made using Python, having an interface in Java. It was developed to simulate the **ideal geometry of a rocket nozzle**. It allows the user to:

- Plot the geometry profile in a 2D graphic and preview the geometry in a 3D plot;
- Calculate the ideal throat radius;
- Plot the ideal expansion ratio to maximize thrust for various chamber pressures;
- Plot performance parameters (Specific Impulse, Thrust, Mach number, etc.) for various O/F ratios;
- Plot exhaust parameters (exhaust temperature and velocity) for different chamber pressures;
- In the 2D plot, the contraction ratio, parabola exit angle, and nozzle length are displayed on the graphic;

> Currently, the simulator has been used to design the nozzle geometry for a **hybrid rocket** using mainly **paraffin wax** as fuel and **N₂O** as oxidizer to compete at EuRock 2025 with Porto Space Team.


### Author:

-[Rafael Lino](https://github.com/rafaelino1707)

### Motivation

The nozzle plays a critical role in rocket engine performance, as it governs the conversion of thermal energy into directed kinetic energy. Due to the limited availability of customizable nozzle geometry simulators, this tool was developed to enable tailored design analysis. It incorporates advanced features not commonly found in standard simulators, such as three-dimensional contour visualization and direct export of discretized profile coordinates for integration with CAD and 3D modeling software.

---
## 2D Geometry Plot

<div align="center">
  <img src="Images/2Dplot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 3D Geometry Preview

<div align="center">
  <img src="Images/3DPlot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## AutoDesk Fusion Model

<div align="center">
  <img src="Images/NozzleGeometryFusion0.png" width="400"/>
</div>

<div align="center">
  <img src="Images/NozzleGeometryFusion1.png" width="363"/>
  <img src="Images/NozzleGeometryFusion2.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Ideal Expansion Ratio

<div align="center">
  <img src="Images/ExpansionRatioPlot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Performance Parameters

<div align="center">
  <img src="Images/PerformanceTable.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Exhaust Velocity

<div align="center">
  <img src="Images/ExhaustVelocityPlot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Exhaust Temperature

<div align="center">
  <img src="Images/ExhaustTemperaturePlot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Thrust Analysis

<div align="center">
  <img src="Images/ThrustPlot.png" width="400"/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
