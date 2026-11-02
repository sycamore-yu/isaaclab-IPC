# PathCover corridor kernel

Source: https://github.com/kunalnk123690/PathCover
Revision: `8723a05d1993bf0199a453b47dc6ee167d493c23`
File: `src/corridor_planning/include/planner/PathCover.hpp`
License: BSD-3-Clause; see `LICENSE`.

Local changes add explicit standard/Qhull includes and replace global `rand()`
with a locally seeded `mt19937` (seed 0) in RISP. Alpha remains 0.01.

IPC retains its path search and corridor scheduling. The binding uses the RISP
single-polytope kernel, not PathCover's whole-path scheduler. It applies the
existing metric clearance to b, then converts Ap <= b to [A, -b].
