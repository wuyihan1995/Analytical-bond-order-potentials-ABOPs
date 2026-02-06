# Analytical-bond-order-potentials-ABOPs
Force field files along with source codes implemented in LAMMPS


To conduct a simulation, you need to place the.cpp and.h files in the LAMMPS source folder (lammps/src). After recompiling, you can use "pair_style tersoff/zbl/my" in the LAMMPS input script. The recommended LAMMPS version is August 2018.

pair_style      tersoff/zbl/my         ##  The short-range interaction is modified via the ZBL correction.
pair_coeff      * *  HfNbZrTaC.tersoff.mod       Hf Nb Zr Ta C     ##  Give element names according to the actual atomic types. 
