CPSea Residue-Break Compatibility Notes
This note summarizes the residue-numbering / backbone-continuity issue found when adapting CPSea complexes to the Proteína-Complexa feature pipeline.
Context
CPSea complexes are stored as receptor–peptide complexes. In the processed files used for our experiments:
Receptor / target chain: usually `A`
Cyclic peptide binder chain: usually `B`
Cyclization bonds: stored separately through `CONECT` records or dataset metadata
Standard peptide backbone connectivity is still represented by consecutive residues and normal `C_i -> N_{i+1}` peptide bonds
Proteína-Complexa pair and sequence features include several quantities that assume a meaningful polymer order, such as relative sequence separation, chain index, and backbone torsion features. Therefore, residue-number jumps and physical chain breaks need to be handled explicitly.
Break-checking rule
For each pair of adjacent residues in file order within a chain, we check:
```text
previous residue C atom  ->  next residue N atom
```
A normal peptide bond has approximately:
```text
C_i -- N_{i+1} ≈ 1.3 Å
```
We use:
```text
C_i -- N_{i+1} > 2.0 Å
```
as a conservative threshold for a physical backbone break.
Each adjacent residue pair is classified as:
```text
REAL_BREAK:
    C_i -> N_next is missing or longer than the cutoff.

NUMBERING_ONLY:
    Residue numbers jump, but C_i -> N_next is still a normal peptide bond.

CLEAN:
    Residue numbers are consecutive and C_i -> N_next is normal.
```
Sample result
We scanned 100 raw gzipped CPSea PDBs and the corresponding 100 preprocessed PDBs.
Dataset	Files	With issues	Clean	REAL_BREAK	NUMBERING_ONLY
`preprocessed_sample100/processed`	100	100	0	1240	0
`CPSea_sample_100` (`.pdb.gz`)	100	100	0	1240	0
The raw and preprocessed results were identical.
Interpretation
The observed receptor residue gaps are real physical discontinuities, not preprocessing artifacts and not pure PDB-numbering quirks.
For example, in `AF-A0A1I1PZC3`:
```text
Chain A
REAL_BREAK: MET A116 -> PHE A119, C-N=3.75 Å
REAL_BREAK: HIS A121 -> HIS A123, C-N=3.52 Å
REAL_BREAK: CYS A126 -> LEU A135, C-N=10.78 Å
...
REAL_BREAK: LEU A300 -> VAL A361, C-N=26.38 Å
```
So target chain `A` should not be treated as one continuous polymer chain. It is better interpreted as a receptor/pocket crop made of multiple continuous segments.
Main issue for Proteína-Complexa features
The dangerous case is:
```text
A126 and A135 are adjacent in the tensor,
but they are not adjacent in the protein backbone.
```
If we naively compute pair features using tensor position, the model may see:
```text
seq_sep(A126, A135) = 1
```
even though the physical peptide bond is broken and the missing segment is absent.
This can corrupt features that rely on polymer adjacency.
Feature-level policy
Target chain / receptor chain
For receptor residues, define a new `segment_id` whenever one of the following is true:
```text
residue number jumps
or C_i -> N_next > 2.0 Å
or C_i / N_next is missing
```
Then use an effective chain identity:
```python
effective_chain_id = (pdb_chain_id, segment_id)
```
For target–target pair features:
```python
same_segment = (
    chain_id[i] == chain_id[j]
    and segment_id[i] == segment_id[j]
)

chain_index = 0 if same_segment else 1
```
This preserves the original Proteína-Complexa convention:
```text
chain_index = 0: same continuous chain-like object
chain_index = 1: different chain-like object
```
For CPSea, “different chain-like object” includes different receptor segments.
Sequence separation
Compute relative sequence separation only within one continuous segment:
```python
if same_segment:
    seq_sep = pos_in_segment[i] - pos_in_segment[j]
    seq_sep = clip(seq_sep, -64, 64)
    seq_sep_feature = one_hot(seq_sep)
else:
    seq_sep_feature = zeros_like_seq_sep_feature
```
Do not assign `seq_sep = 1` across a receptor break merely because two residues are adjacent in the cropped tensor.
Backbone torsions
Backbone torsions should be valid only when all required neighboring residues are in the same segment and required atoms exist.
For residue `i`:
```text
phi_i:
    valid only if residue i-1 and i are in the same segment

psi_i:
    valid only if residue i and i+1 are in the same segment

omega_i:
    valid only if residue i and i+1 are in the same segment
```
Invalid torsions should be encoded as a null feature. If torsions are one-hot binned, the null feature should be an all-zero vector, not bin `0`.
```python
invalid_torsion_onehot = torch.zeros(num_torsion_bins)
```
Bin `0` is a real angle bin, not a missing-value marker.
Side-chain torsions
Side-chain torsions usually do not depend on inter-residue backbone continuity. They should be marked invalid only when:
```text
the residue type does not have that chi angle
or required side-chain atoms are missing
```
Binder chain
The cyclic peptide binder chain, usually chain `B`, should normally be a contiguous peptide backbone. CPSea filters peptide candidates to have contiguous residue indices before cyclization.
Therefore, expected healthy binder-chain counts are:
```text
B_REAL_BREAK = 0
B_NUMBERING_ONLY = 0
```
If `B_REAL_BREAK > 0`, this indicates a serious issue for training because the binder backbone itself is broken.
The cyclization bond should not be confused with normal linear peptide-bond adjacency. The binder should still be indexed linearly:
```text
0, 1, 2, ..., L_b - 1
```
The cyclization is an extra typed edge:
```python
cyclization_edge[i, j] = True
cyclization_type[i, j] = "mainchain" / "disulfide" / "isopeptide" / ...
```
It should not overwrite the usual linear `seq_sep`.
Binder residue indexing
For generated binders, original PDB residue numbers do not exist. The model generates residue slots:
```text
0, 1, 2, ..., L_b - 1
```
or equivalently:
```text
1, 2, ..., L_b
```
depending on code convention.
So binder–binder sequence separation should use generated slot indices, not source PDB residue numbers.
For training, even if a CPSea binder appears as:
```text
B237, B238, ..., B249
```
it should be reset to:
```text
0, 1, ..., 12
```
or:
```text
1, 2, ..., 13
```
before constructing binder features.
Absolute positional embedding
In the current Proteína-Complexa code, absolute residue index embedding uses `residue_pdb_idx` if present, but first shifts each row so that the first valid residue starts at `1`.
Conceptually:
```python
idx = residue_pdb_idx
idx = idx - idx[:, :1] + 1
```
Therefore:
```text
B237, B238, ..., B249
```
and:
```text
1, 2, ..., 13
```
produce the same absolute positional embedding after this reset.
For pairwise `seq_sep`, using raw PDB numbers or reset numbers gives the same result only if the numbering is contiguous, because pairwise differences are translation-invariant:
```text
(249 - 237) == (13 - 1)
```
But for CPSea preprocessing, reset binder numbering is still cleaner and safer.
Script
Use `scripts/check_pdb_residue_jumps.py` to scan either a single PDB/PDB.GZ file or a directory.
Example:
```bash
python scripts/check_pdb_residue_jumps.py CPSea_sample_100 --pattern "*.pdb.gz"
```
The script reports total counts and separate binder-chain counts:
```text
REAL_BREAK
NUMBERING_ONLY
B_REAL_BREAK
B_NUMBERING_ONLY
```
The default binder chain is `B`. To override:
```bash
python scripts/check_pdb_residue_jumps.py data_dir --binder-chain L
```
Recommended preprocessing changes
Preserve original residue numbers for traceability.
Reset binder residue positions to `0..L_b-1` or `1..L_b`.
Split receptor/target chain into continuous `segment_id`s.
Treat different receptor segments as different chain-like objects in `chain_index`.
Null/mask target sequence-separation features across segment boundaries.
Null/mask backbone torsions across segment boundaries.
Keep side-chain torsions valid if the residue itself has the required atoms.
Parse cyclization bonds separately from `CONECT` or CPSea metadata.
Add typed cyclization pair/edge labels instead of forcing cyclicity into `seq_sep`.
Bottom line
CPSea receptor chains are often discontinuous receptor/pocket crops. This is compatible with the Proteína-Complexa architecture only if target segments are handled explicitly.
The binder chain should remain contiguous in the standard peptide backbone. Cyclization should be represented as an additional typed edge, not as a normal sequence-adjacent peptide bond.