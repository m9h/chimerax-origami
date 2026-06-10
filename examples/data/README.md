# Example data — provenance

| File | Source | Notes |
|---|---|---|
| `demo_design.json` | this repo | tiny hand-made contactmap-format design for the `.cxc` demo |
| `demo_assembly.npz` | this repo | synthetic oxDNA-like folding trajectory (occupancy) for `assembly_msm` |
| `simple42legacy.json` | [douglaslab/cadnano2](https://github.com/douglaslab/cadnano2) `tests/functionaltestinputs/` | minimal legacy-format design; golden-file unit test |
| `Nature09_monolith.json` | [douglaslab/cadnano2](https://github.com/douglaslab/cadnano2) `tests/functionaltestinputs/` | **Douglas et al., *Nature* 459, 414 (2009)** honeycomb monolith — p7560 scaffold (7560 nt), 144 staples, 18 helices; real-design importer validation |
| `m13mp18.txt` | [lcbb/pyDAEDALUS](https://github.com/lcbb/pyDAEDALUS) | M13mp18 scaffold sequence (7249 nt); scorer validation (test 1) |
| `caca.json` | [oxDNA](https://github.com/lorenzo-rovigatti/oxDNA) `examples/NEW_RELAX_PROCEDURE/` | small cadnano origami (290-nt scaffold, 5 staples, 608 nt) |
| `caca.json.top` | tacoxDNA output for `caca.json` (same source) | oxDNA topology; importer cross-validation (test 2) |

The two cadnano files are canonical community test designs redistributed from
the cadnano2 repository (these same files also ship with cadnano2.5 and appear
in oxDNA tutorials). They are used here only as importer test fixtures.
