# Roadmap

- **MIND path on a released compiler.** The MIND kernel currently needs a compiler
  built from the compiler repository's main branch (runtime-indexed tensor
  parameters). When the next release carries them, pin it, run `tools/infer_mind.py`
  in CI, and gate it on the pinned trace checksum like the C++ paths. Done when the
  MIND row in the results table is produced by CI, bit-identical to the reference.
- **Native backend.** When the compiler's native backend admits the tensor C ABI,
  the MIND path builds through it instead of the MLIR route; same checksum gate.
