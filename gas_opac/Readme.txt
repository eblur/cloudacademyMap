*****DISCLAIMER****

The cross section database enclosed is a work in progress. The method used to compute these cross sections is currently unpublished (MacDonald, in prep), but the resulting cross sections have been benchmarked against other groups (e.g. EXOMOL, NASA Ames). They should be considered experimental at this stage, and I would appreciate if you do not distribute them beyond this collaboration without prior approval until further testing has been completed. If you would like to use these cross sections for other projects before the database is published, please drop me an email at r.macdonald@ast.cam.ac.uk and I'll be happy to provide you with updates.

*****CONTENTS*****

The full details underlying the cross sections will be laid out elsewhere, but the main characteristics are as follows:

*Ions included: H3+, Fe+, Ti+
*Atoms included: Na, K, Li, Rb, Cs, Fe, Ti
*Molecules included: H2, H2O, CH4, NH3, HCN, CO, CO2, C2H2, H2S, N2, O2, O3, OH, NO, SO2, PH3, TiO, VO, ALO, 
                     SiO, CaO, TiH, CrH, FeH, ScH, AlH, SiH, BeH, CaH, MgH, LiH, SiH, CH, SH, NH

*All species are broadened by H2 and He in an 85% / 15% ratio (accounting for angular momentum quantum number J dependance where available).

*Cross sections are calculated on a uniform wavenumber grid from 200 to 25000 cm^-1 (i.e. 50 um -> 0.4 um) with a spacing of 0.01 cm^-1 (R~10^6).

*Each cross section calculated at 162 pressure-temperaure points:
log(P/bar) = -6.0 -> 2.0 (1 dex spacing)
T/K = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500]

*All linelists are the latest available in the literature (mostly ExoMol).

*For cross sections, the stored values are log10(cross section / m^2) (species, nu, P, T).

***NEW***

*Collisionally-induced absorption (CIA) is only a function of T and nu, and is tabulated from HITRAN.

*For CIA, the stored values are log10(binary cross section / m^5) (cia_pair, nu, T).

*****USAGE*****

See the python script 'opacity_demo.py' in this folder for an example of how to open the database, select a given species, and plot it at a given temperature and pressure.

Note that the pre-computed cross section arrays are extremely large (9x18x2480001 elements for each species), so running on a computer with >8 GB RAM is generally advised.

*****QUESTIONS/FEEDBACK*****

Any questions, comments, or feedback? New atoms, molecules, or ions you would like to see included?
Please drop a message to Ryan MacDonald @ r.macdonald@ast.cam.ac.uk
