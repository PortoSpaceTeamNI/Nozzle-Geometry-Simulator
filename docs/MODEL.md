# Model reference

This document records what the simulator calculates and which quantities come from
RocketCEA. It is deliberately explicit so that future optimizers can call the same
model without duplicating equations.

## Inputs

### Operating point

- Chamber pressure, `Pc` [bar]
- Oxidizer-to-fuel mixture ratio, `O/F` [-]
- Ambient pressure, `Pamb` [atm]
- Exit-to-throat area ratio, `epsilon = Ae/At` [-]

### Pre-sizing

- Throat radius, `Rt` [m]
- Chamber diameter [m]

### Optimized contour variables

- Expansion ratio, `epsilon = Ae/At` [-]
- Bell fraction, `Kb = Lparab/Lcone` [-]
- Initial divergent-wall angle, `theta_in` [deg]

The convergent straight-line angle, `theta_sub`, is a user-selected fixed input
during optimization.

The 15-degree reference cone angle `alpha` remains fixed and defines the reference
length.

## Integrated contour optimization

The desktop optimizer holds `Pc`, O/F, `Pamb`, pre-sizing inputs and `alpha` fixed.
It also holds `theta_sub` fixed. It varies expansion ratio, bell fraction `Kb` and
`theta_in`.
Every candidate calls the same
geometry, CEA, quasi-1D and boundary-layer modules used by the plots. Invalid contours
receive zero merit.

The optimized objective is the effective ambient thrust coefficient:

```text
eta_momentum = eta_divergence eta_BL
eta_divergence = (1 + cos(theta_out))/2
Cf_momentum = CFcea returned by RocketCEA
Cf_pressure = epsilon (Pe - Pamb)/Pc
Cf_friction = integral(2 pi r tau_w dx)/(Pc At)
Cf_effective = eta_momentum Cf_momentum + Cf_pressure - Cf_friction
```

Only the momentum contribution is reduced by divergence and boundary-layer losses.
The pressure contribution retains its sign. Candidates classified as separated by
RocketCEA receive zero fitness because the exit-plane pressure-thrust expression no
longer represents the separated flow.

The interface reports a loss allocation in thrust-coefficient units. Divergence is
`(1-eta_divergence) Cf_momentum`; BL displacement is
`(eta_divergence-eta_momentum) Cf_momentum`; wall friction is `Cf_friction`; and
ambient mismatch is `max(-Cf_pressure, 0)`. A positive pressure term is a gain and
is excluded from percentage loss shares.

The optimizer exposes two candidate evaluators without changing its chromosome or
operators:

- `blimp`: the BLIMP-lite/Cebeci-Smith profile marcher described below;
- `quick`: a screening-only momentum-integral model with an imposed 1/7-power/Walz
  profile and `Cf = 0.0592/Re_s,ref^0.2` using Eckert reference temperature.

Every geometry returned by a Quick search is regenerated with the full BLIMP-lite
path before its final plots and reported validation value are shown.

## RocketCEA properties

RocketCEA receives `Pc`, `O/F` and `epsilon`. The simulator requests chamber,
throat and exit values using explicit units:

- temperature [K];
- molecular weight [g/mol];
- specific-heat ratio `gamma` [-];
- heat capacity [J/(kg K)];
- viscosity [Pa s];
- conductivity [W/(m K)];
- Prandtl number [-];
- characteristic velocity `c*` [m/s];
- exit Mach number and `Pc/Pe`;
- ideal momentum and ambient thrust coefficients;
- ideal `epsilon` satisfying `Pe = Pamb` and the ambient operating mode.

The custom paraffin fuel card is defined once in `nozzle_simulator/cea.py`.

## Geometry

The contour follows the original `NozzleGeometry.py` construction:

1. convergent straight line;
2. circular subsonic throat arc of radius `1.5 Rt`;
3. circular supersonic throat arc of radius `0.4 Rt`;
4. quadratic bell ending at `Re = Rt sqrt(epsilon)`.

The reference cone length is

```text
Lcone = (Re - Rt) / tan(alpha)
```

and the quadratic part restores the legacy definition

```text
Lparab = bell_fraction * Lcone
```

Bell fraction is the single parameter controlling divergent length in manual
generation and optimization. For `r(x) = a x^2 + b x + c`, the three coefficients
enforce

```text
r(x_in)   = r_in
r'(x_in)  = tan(theta_in)
r(x_exit) = r_exit
```

The exit wall angle is a derived result:

```text
theta_out = atan(2 a x_exit + b)
```

It still affects the divergence factor directly and the integral boundary-layer model
through the complete wall contour. It is deliberately not an independent gene.
Candidates whose radius decreases or whose derived exit angle is negative are rejected.

`theta_sub` changes the convergent straight-line length, wall area and subsonic area
gradient. Both BLIMP-lite and Quick march over the complete convergent contour, so
its displacement and wall-friction effects still enter every candidate evaluation.
It is not optimized because there is currently no separate contraction-loss,
chamber-corner, combustion-stability or manufacturability closure capable of giving
a trustworthy interior optimum.

## Flow, thermal and boundary layer

The internal profile is a quasi-one-dimensional isentropic estimate. Mach is found
from the area-Mach relation on the subsonic or supersonic branch. Gamma transitions
between the CEA chamber, throat and exit values.

The gas-side wall is adiabatic. Its temperature is not an independent input:

```text
r = Pr_e^(1/3)
Tw = Tr = Taw = Te [1 + r (gamma_e - 1) Me^2 / 2]
qw = 0
```

`Pr_e` is interpolated locally from the chamber, throat and exit RocketCEA
properties. The Thermal tab retains the Bartz heat-transfer coefficient as a
diagnostic for a future conjugate wall/cooling model, but it does not invent a heat
flux from an unknown fixed wall temperature.

The boundary-layer module is a non-reacting, attached-flow **BLIMP-lite** marcher.
It is inspired by the JANNAF BLIMP-J integral-matrix procedure but is not a copy of
the historical program. It resolves a wall-normal velocity profile and enforces the
compressible axisymmetric thin-layer equations:

```text
d(r rho u)/ds + d(r rho v)/dy = 0

rho u du/ds + rho v du/dy
    = -dpe/ds + (1/r) d[r (mu + mut) du/dy]/dy
```

The outer pressure gradient comes from the quasi-1D edge flow. Density and
molecular viscosity vary across the layer. The adiabatic Walz relation closes the
temperature profile:

```text
T(y) = Taw - (Taw - Te) [u(y)/Ue]^2
```

Turbulence is closed with the two-layer Cebeci-Smith algebraic eddy viscosity:

```text
l = kappa y [1 - exp(-y+/A+)]
mut,inner = rho l^2 |du/dy|
mut,outer = 0.0168 rho_e Ue delta* F_Kleb(y)
mut = min(mut,inner, mut,outer)
```

with `kappa=0.40` and `A+=26`. At the wall `mut=0`, so wall shear and local skin
friction follow from the resolved gradient rather than a flat-plate Cf correlation:

```text
tauw = muw (du/dy)w
Cf = 2 tauw / (rho_e Ue^2)

delta* = integral[1 - rho u/(rho_e Ue)] dy
theta  = integral[rho u/(rho_e Ue) (1-u/Ue)] dy
```

The momentum equation is marched implicitly on 61 streamwise stations and 41
wall-normal points clustered toward the wall. Picard iteration updates density,
viscosity, wall units and eddy viscosity. The resulting profiles are interpolated
back onto the complete contour grid. The selected grid was compared with a 91 by 57
grid in the default case: exit Cf and integrated friction Cf changed by less than
0.5%, while exit displacement thickness changed by about 4%.

Displacement blockage is applied through `r_eff = r - delta*`. Wall shear is also
integrated as an independent axial thrust loss:

```text
Ffric = integral 2 pi r tauw dx
CFfric = Ffric/(Pc At)
CFeffective = eta_div eta_blockage CFmomentum + CFpressure - CFfric
```

The implementation basis is the [BLIMP-J user's manual, NASA-CR-144046](https://ntrs.nasa.gov/citations/19760066866)
and its [JANNAF turbulence-model verification report](https://ntrs.nasa.gov/citations/19770017245),
which recommends Cebeci-Smith for the tested liquid-rocket boundary layers.

## Axisymmetric MOC and Sauer initialization

The optional prescribed-wall MOC uses the CEA throat molecular weight to obtain the
specific gas constant and holds the CEA throat gamma fixed throughout the perfect-gas
characteristic march. The low-order Sauer initializer uses the existing convergent
throat radius `Rc,sub = 1.5 Rt`:

```text
alpha = sqrt(2 / [(gamma + 1) Rc,sub Rt])
eta   = (gamma + 1) alpha Rt^2 / 8
a*    = sqrt(2 gamma R T0 / (gamma + 1))
```

`Nr` is the number of discrete points from the symmetry axis to the throat wall on
the initial-data line; the same radial count is used by the fixed-x MOC grid. The
curved Sauer line and its state are retained in the result and export. Since the
current body-fitted marcher requires a complete constant-x first plane, the analytic
Sauer field is sampled on a wholly supersonic axial plane. This projection is marked
experimental and must not be considered verified while the mass-flow residual exceeds
0.5%. The interface retains the previous quasi-1D initializer solely as a comparison
and timing reference.

## Limitations

- This is an engineering design tool, not a CFD solver.
- The optional MOC resolves a preliminary multidimensional inviscid exit field but
  does not yet capture shocks or separation and has not met its mass-conservation
  verification target on the tested grids.
- BLIMP-lite assumes fully turbulent attached flow, no wall mass transfer and an
  adiabatic wall. It does not currently predict transition, relaminarization or
  separation.
- The inlet boundary-layer profile is an engineering initialization; chamber history
  upstream of the modelled contour is not known.
- The energy field uses the adiabatic Walz relation instead of BLIMP-J's full
  chemically reacting energy/species system.
- Cebeci-Smith is an algebraic equilibrium closure. Strong non-equilibrium turbulence,
  separation and shocks require RANS/CFD or experimental calibration.
- Bartz `hg`, wall shear and displacement require validation for the propellants and
  operating range of interest.
- Always compare final designs against RocketCEA/CEA reports, CFD and hot-fire data.
