OpenMM MM and ML/MM + MMPBSA.py Workflow
========================================

This repository contains a small, self-contained example workflow for running
OpenMM MM or ML/MM simulations and then computing binding free energies with
MMPBSA.py.

Contents
--------
- mlmm_gbsa.yml : Conda environment spec (OpenMM/OpenFF/openmm-ml/AmberTools)
- mm.py         : Pure MM simulation script (OpenMM + OpenFF)
- mlmm.py       : ML/MM simulation script (OpenMM + OpenFF + openmm-ml)
- gbsa.py       : Converts OpenMM outputs to Amber and runs MMPBSA.py
- gbsa.sh       : Convenience wrapper for gbsa.py
- protein.pdb   : Example protein input
- ligand.sdf    : Example ligand input

Quick Start
-----------
1) Create and activate the environment

   conda env create -f mlmm_gbsa.yml
   conda activate mlmm_gbsa

2) Prepare inputs

   - Put your protein in protein.pdb and ligand in ligand.sdf
   - Check the force fields in mm.py / mlmm.py

3) Run a simulation

   python mm.py / python mlmm.py

   Outputs are written to ./md_output/:
   - equil_npt.pdb
   - simulation.xtc

4) Run MMPBSA.py

   bash gbsa.sh

   Results are written to ./gbsa_results/:
   - FINAL_RESULTS_MMPBSA.dat

Notes
-----
- gbsa.sh passes --datadir ./md_output. Update it if your outputs differ.
- If the ligand residue name in the PDB is not UNK, set --ligand-resname.
- Make sure --ligand-ff in gbsa.sh matches the force field used in the MD run.
- Use --method gb, pb, or both in gbsa.sh to control the solvent model.
- This conda environment only support AceFF and AIMNet2, for other MLFF models such as MACE should adjust the installed packages.
