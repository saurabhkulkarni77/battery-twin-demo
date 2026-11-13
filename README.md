# Battery Multi-Scale Digital Twin — Concept Demo

Interactive concept demo for the proposal "Manufacturing Memory Across Scales and Stages"
(Li-ion electrode manufacturing, BEACONS pilot line, UT Dallas).

One screen, four linked scales — electrode microstructure, cell, 96-cell pack, service
life — driven by six process knobs (binder adhesion, binder content, drying rate,
calendering pressure, particle size, charge rate). All values are computed live in the
browser by a small surrogate chain (see the model notes in index.html / app.js); in the
full system each stand-in is replaced by LAMMPS (MD), Yade/LIGGGHTS (DEM),
FEniCSx + TauFactor (FEM), PyBaMM (cell), and physics-informed neural operators
exported to ONNX. Simulated data only — a starting concept for discussion.
