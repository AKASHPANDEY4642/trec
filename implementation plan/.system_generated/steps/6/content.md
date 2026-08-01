Title: Live Content

Description: Fetched live

Source: https://raw.githubusercontent.com/TREC-RAG/trec-rag-skills/main/skills/trec-rag-2026-track-guidelines/references/retrieval-task.md

---

# Retrieval Task (`R`)

Use this reference when building, explaining, or validating the TREC RAG 2026 Retrieval task.

## Task Summary

- **Given**: a list of narratives and access to the ClimbMix collection through the Pyserini REST API or a custom retrieval system.
- **Task**: return a ranked list containing all and only the ClimbMix documents the system predicts are relevant to the narrative and useful as evidence for answer generation.
- **Depth**: choose the submitted depth `k` separately for each narrative. There is no organizer-supplied fixed cutoff to fill.

## Input Format: Narratives

Use the official shared test-narrative file described in [test-data.md](test-data.md). Narratives are provided as TSV in `trec_rag_2026_queries.tsv`. Each line contains the narrative ID and narrative, separated by a tab.

```tsv
rag2026-37	I work for a New York City council member whose district has a lot of transit riders but also some small businesses worried about delivery costs. Can you help me understand whether congestion pricing is a credible and fair way to fund the MTA? What should we weigh about the revenue promise, who pays, who benefits, environmental tradeoffs in places like the Bronx and New Jersey, and whether the MTA and Albany can be held accountable for actually spending the money on reliable service instead of repeating past mistakes?
```

Required fields:

- First column: narrative identifier. Preserve this exactly in all outputs.
- Second column: narrative, usually a two- to three-sentence description of the information need. Use it as the default initial retrieval query unless the system intentionally performs query rewriting or decomposition internally. For `RAG` output, copy this value exactly into `metadata.narrative`.

## Input Format: Documents

For baseline systems, retrieve ClimbMix documents from the Pyserini REST API. The configured index name is:

```text
climbmix-400b
```

Search returns a response with a `candidates` array. Each candidate represents one retrieved ClimbMix document:

```json
{
  "api": "v1",
  "index": "climbmix-400b",
  "query": { "text": "congestion pricing MTA funding accountability" },
  "candidates": [
    {
      "docid": "shard_00459_61697",
      "rank": 1,
      "score": 12.483799934387207,
      "doc": "..."
    }
  ]
}
```

Document fetch by ClimbMix document ID returns one document wrapper:

```json
{
  "api": "v1",
  "index": "climbmix-400b",
  "docid": "shard_00459_61697",
  "doc": "..."
}
```

Document field rules:

- `docid`: ClimbMix document ID to use in submissions.
- `rank`: returned rank for a search candidate.
- `score`: retrieval score for a search candidate.
- `doc`: ClimbMix document contents. The Pyserini REST API schema allows this payload to be a string, object, array, number, boolean, or null depending on index contents and `parse` behavior. Current ClimbMix search responses commonly return `doc` as a string containing the doc

