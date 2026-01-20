# RNA Structure Metadata (v330) - Column Descriptions

The `rna_metadata.csv` contains metadata for RNA or hybrid RNA-DNA chains extracted from the Protein Data Bank (PDB) for the date range 1978-01-01 to 2025-12-17.

## Identifiers

- **target_id**: Unique identifier combining PDB ID and chain (`{pdb_id}_{auth_chain_id}`)
- **pdb_id**: Protein Data Bank structure identifier
- **chain_id**: Chain identifier used in the structure file
- **auth_chain_id**: Author-assigned chain identifier
- **entity_id**: Entity identifier from the PDB entry
- **entity_type**: Type of molecular entity (e.g., "rna", "na_hybrid")

## Sequence Information

- **sequence**: Canonical RNA sequence (using standard nucleotide codes)
- **canonical_sequence**: Colon-separated canonical nucleotide sequence
- **full_sequence**: Colon-separated sequence including modified nucleotides (e.g., 2MG, PSU, 5MC)
- **sequence_expected**: Expected sequence from PDB annotation (colon-separated)
- **sequence_observed**: Observed sequence in structure coordinates (colon-separated, '?' for unresolved residues)

## Quality Metrics

- **unmapped_canonical**: Boolean indicating if canonical sequence mapping failed
- **unmapped_nakb**: Boolean indicating if mapping from modified to canonical nucleotide using NAKB failed 
- **mapped_fraction**: Fraction of residues that could not be mapped to standard types
- **undefined_residues**: Boolean indicating presence of undefined residues (such as N)
- **unexpected_residues**: Boolean indicating presence of unexpected residues (other than ACGU, N and X)
- **nonstandard_residues**: Boolean indicating presence of nonstandard/modified residues 
- **missing_residues**: Boolean indicating if any residues are missing from coordinates

## Composition

- **composition_total_mass**: Total molecular mass (Da)
- **composition_rna_mass**: Mass contributed by RNA entities (Da)
- **composition_rna_fraction**: Fraction of total mass from RNA
- **composition_na_hybrid_mass**: Mass contributed by DNA/RNA hybrid entities (Da)
- **composition_na_hybrid_fraction**: Fraction of total mass from NA hybrids

## Assembly Information

- **assembly_num_assemblies**: Number of biological assemblies defined
- **assembly_assembly_defined**: Boolean indicating if biological assembly is defined
- **assembly_has_symmetry_operators**: Boolean indicating presence of symmetry operators
- **assembly_rna_entities**: Number of RNA entities in the assembly
- **assembly_rna_copies**: Number of RNA copies in the assembly
- **assembly_na_hybrid_entities**: Number of NA hybrid entities in the assembly
- **assembly_na_hybrid_copies**: Number of NA hybrid copies in the assembly

## Structure Metadata

- **temporal_cutoff**: Initial release date of the structure (YYYY-MM-DD)
- **resolution**: Structure resolution if reported
- **method**: Experimental method (e.g., X-ray, NMR, EM)
- **title**: Structure title from PDB entry
- **keyword_ribosome**: Boolean indicating if structure is annotated as ribosome-related

## Grouping and Clustering

- **group_id**: Group ID for sequences sharing ≥90% coverage and 100% identity
- **seq_group_id**: Group ID for sequences with 100% identity
- **mmseqs_0.300** through **mmseqs_0.950**: MMseqs2 cluster IDs at different sequence identity thresholds (30% to 95%)

## External Database Mappings

- **rna3ddb_component_id**: RNA 3D Database component identifier
- **rna3ddb_cluster_id**: RNA 3D Database cluster identifier
- **rna3dhub_id**: RNA 3D Hub non-redundant list identifier

## Structuredness Metrics

Structuredness measures the fraction of residues with nearby neighbors (within 12Å of C1' atoms):

- **inter_chain_structuredness**: Fraction of residues within 12Å of a different RNA chain
- **intra_chain_structuredness**: Fraction of residues within 12Å of the same chain (>4 residues away in sequence)
- **total_structuredness**: Fraction of residues within 12Å of any other residue
- **inter_chain_structuredness_adjusted**: Inter-chain metric adjusted for unresolved residues (× fraction_observed)
- **intra_chain_structuredness_adjusted**: Intra-chain metric adjusted for unresolved residues (× fraction_observed)
- **total_structuredness_adjusted**: Total metric adjusted for unresolved residues (× fraction_observed)

## Length Metrics

- **length**: Total sequence length
- **length_observed**: Number of residues with resolved coordinates (excluding '?' in sequence_observed)
- **length_expected**: Expected number of residues based on sequence annotation
- **fraction_observed**: Fraction of expected residues with resolved coordinates (length_observed / length_expected)
