

![VAWT850](/VAT_Flow_Example.png?raw=true "Example")

# VAT-ASM
Repository containing the python code used in the paper "Numerical Modelling of Vertical Axis Turbines using the Actuator Surface Model".

This tool can be used for quick predesign of vertical axis turbines (VAT). It combines a 2D computational fluid dynamics (CFD) simulation with the classical blade element theory (BEM). Due to the fact that no blades have to be resolved directly, the computation time can be reduced considerably. In addition, analytical higher order corrections can be applied to approximate dynamic blade stall and 3D effects. The script allows the user to use his own tabulated airfoil coefficients and to configure the turbine parameters himself. For parallel computation, the Python functions are accelerated with the just-in-time compiler [Numba](https://github.com/numba/numba).

*Keywords:* Actuator surface model, BEM, Blade element theory, CFD, VAWT, Water turbine, DMST

## Prerequisites

Mainly only the standard packages are needed, which are already included in most Python distributions, such as Anaconda.
* Numba
* Scipy
* Numpy

## Usage

* Modify the config .ini files as desired
* Just run *VAT_ASM_Main* as a script


## Paper

The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0889974621001018).

## Contact

If you want to contact me you can reach me at <Jan-Philipp.Kueppers@uni-siegen.de>.


