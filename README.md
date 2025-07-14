# Rupture de barrage 1D — Saint-Venant

Ce projet simule la rupture d’un barrage en 1D avec les équations de Saint-Venant, via un schéma de Rusanov. Trois versions sont disponibles :

- `dam_break_rusanov.c` — séquentielle  
- `dam_break_rusanov_openmp.c` — OpenMP (mémoire partagée)  
- `dam_break_rusanov_mpi.c` — MPI (mémoire distribuée)

## Compilation

```bash
gcc -o seq dam_break_rusanov.c 
gcc -fopenmp -o openmp dam_break_rusanov_openmp.c 
mpicc -o mpi dam_break_rusanov_mpi.c
