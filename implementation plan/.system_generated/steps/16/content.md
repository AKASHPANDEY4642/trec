Title: Live Content

Description: Fetched live

Source: https://raw.githubusercontent.com/TREC-RAG/trec-rag-skills/main/skills/trec-rag-2026-track-guidelines/references/test-data.md

---

# TREC RAG 2026 Test Narratives

Sources checked July 17, 2026:

- Release announcement: https://x.com/TREC_RAG/status/2074513634043064419
- Track homepage: https://trec-rag.github.io/
- Data repository: https://github.com/TREC-RAG/trec-rag-data
- Official narrative file: https://github.com/TREC-RAG/trec-rag-data/blob/main/trec-rag-2026/test-data/trec_rag_2026_queries.tsv

## Summary

The official TREC RAG 2026 test narratives have been released. The same test-data file is the input for both the Retrieval (`R`) and Retrieval-Augmented Generation (`RAG`) tasks.

Use the official file from the [`TREC-RAG/trec-rag-data`](https://github.com/TREC-RAG/trec-rag-data) repository:

```text
trec-rag-2026/test-data/trec_rag_2026_queries.tsv
```

The file is named:

```text
trec_rag_2026_queries.tsv
```

## Contents and Schema

The released file contains 119 narratives, with narrative identifiers from `rag2026-0` through `rag2026-118`. It has no header row.

Each line contains exactly two tab-separated fields:

```text
narrative_id<TAB>narrative
```

For example:

```tsv
rag2026-37	I work for a New York City council member whose district has a lot of transit riders but also some small businesses worried about delivery costs. Can you help me understand whether congestion pricing is a credible and fair way to fund the MTA? What should we weigh about the revenue promise, who pays, who benefits, environmental tradeoffs in places like the Bronx and New Jersey, and whether the MTA and Albany can be held accountable for actually spending the money on reliable service instead of repeating past mistakes?
```

- `narrative_id`: the official identifier. Preserve it exactly in every task output.
- `narrative`: the official long-form description of the information need. Preserve it exactly when a task output requires the narrative.

Parse the file as TSV, not as arbitrary whitespace-separated text. Narratives contain many spaces and may contain punctu

