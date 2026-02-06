# Analytical-bond-order-potentials-ABOPs

Force field files along with source codes implemented in LAMMPS

To conduct a simulation, you need to place the.cpp and.h files in the LAMMPS source folder (lammps/src). After recompiling, you can use "pair_style tersoff/zbl/my" in the LAMMPS input script. The recommended LAMMPS version is August 2018.


//-------------------------------------------------------------------------------------------//
##  The short-range interaction is modified via the ZBL correction.
pair_style      tersoff/zbl/my         
##  Give element names according to the actual atomic types. 
pair_coeff      * *  HfNbZrTaC.tersoff.mod       Hf Nb Zr Ta C    




Citation: Yihan Wu, Wenshan Yu, Shengping Shen,
Developing an analytical bond-order potential for Hf/Nb/Ta/Zr/C system using machine learning global optimization,
Ceramics International,
Volume 49, Issue 21,
2023,
Pages 34255-34268,
DOI: 10.1016/j.ceramint.2023.08.139.
