This is the original version of MUSIC, for v2, please go to [this bitbucket repo](https://github.com/cosmo-sims/MUSIC)

MUSIC - multi-scale cosmological initial conditions
===================================================

MUSIC is a computer program to generate nested grid initial conditions for
high-resolution "zoom" cosmological simulations. A detailed description
of the algorithms can be found in [Hahn & Abel (2011)][1]. You can
download the user's guide [here][3], or [read the Wiki](https://bitbucket.org/ohahn/music/wiki/Home) instead. Please consider joining the
[user mailing list][2].

Current MUSIC key features are:

- Supports output for RAMSES, ENZO, Arepo, Gadget-2/3, ART, Pkdgrav/Gasoline 
and NyX via plugins. New codes can be added.

- Support for first (1LPT) and second order (2LPT) Lagrangian perturbation 
theory, local Lagrangian approximation (LLA) for baryons with grid codes.

- Pluggable transfer functions, currently CAMB, Eisenstein&Hu, BBKS, Warm 
Dark Matter variants. Distinct baryon+CDM fields.

- Minimum bounding ellipsoid and convex hull shaped high-res regions supported 
with most codes, supports refinement mask generation for RAMSES.

- Parallelized with OpenMP
    
- Requires FFTW (v2 or v3), GSL (and HDF5 for output for some codes)

## Building MUSIC
While we still supply the old Makefile, using CMake is now the preferred way of building. 
CMake use out-of-source build, i.e. you create a build directory, and then configure the code using CMake. Inside the `music` directory, do
```
  mkdir build
  cd build
  ccmake ..
  make -j
```
to configure the code (you will se a menu), and then start a parallel compilation. If CMake has trouble finding your FFTW or HDF5 installation,
you can add hints as follows
```
  FFTW3_ROOT=<path> HDF5_ROOT=<path> ccmake ..
```
If you want to build on macOS, then it is strongly recommended to use GNU (or Intel) compilers instead of Apple's Clang. Install them e.g. via homebrew and then configure cmake to use them instead of the macOS default compiler via
```
  CC=gcc-9 CXX=g++-9 ccmake ..
```
This is necessary since Apple's compilers haven't supported OpenMP for years.


## Running

There is an example parameter file 'example.conf' in the main directory. Possible options are explained in it, it can be run
as a simple argument, e.g. from within the build directory:

```
  ./MUSIC ../ics_example.conf
```

For DMO initial conditions whose displacement uses a model-specific density
transfer, the optional `[setup]` key `vfact_scale` multiplies MUSIC's
background velocity growth factor without changing the density or particle
positions. Its default is `1.0`; the value should be derived from the same
linear solver and epoch as the density transfer.

For DMO 2LPT initial conditions, `[setup] dmo_velocity_source` selects the
velocity source:

- `transfer` (default) constructs the first-order velocity potential directly
  from the input plugin's mass-weighted `vtotal` transfer and then adds MUSIC's
  high-redshift 2LPT velocity term.
- `density_2lpt` retains the legacy path in which both the first- and
  second-order velocities are derived from the density-transfer potential.

The `transfer` mode retains the density-transfer potential for particle
displacements, requires a plugin with velocity columns, should use
`vfact_scale = 1`, and supports both uniform and nested-grid ICs. MUSIC stops
with an explicit error if `transfer` is selected for an input with no velocity
field. For backward compatibility, an old configuration that omits the key
and uses a plugin without velocities defaults to `density_2lpt` with a
warning; velocity-capable inputs default to `transfer`.


## Disclaimer

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. By downloading and using MUSIC, you 
agree to the LICENSE, distributed with the source code in a text 
file of the same name.



[1]: http://arxiv.org/abs/1103.6031
[2]: https://groups.google.com/forum/#!forum/cosmo_music
[3]: https://bitbucket.org/ohahn/music/downloads/MUSIC_Users_Guide.pdf
