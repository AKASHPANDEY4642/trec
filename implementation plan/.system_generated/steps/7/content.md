Title: Live Content

Description: Fetched live

Source: https://raw.githubusercontent.com/TREC-RAG/trec-rag-skills/main/skills/trec-rag-2026-track-guidelines/SKILL.md

---

---
name: trec-rag-2026-track-guidelines
description: Use when discussing, building, validating, or explaining the TREC RAG 2026 track, systems, baselines, participation, or submissions. This skill covers the 2026 track overview, released test and development data, public status, organizers, participation guidance, Retrieval and Retrieval-Augmented Generation tasks, ClimbMix/Pyserini REST retrieval defaults, required input and output formats, citation rules, and validation checks for agent-created TREC RAG 2026 runs.
metadata:
  version: v0.6.0
---

# TREC RAG 2026 Track Guidelines

Use this skill when answering conversational questions about the TREC RAG 2026 track, orienting new participants, preparing a TREC RAG 2026 submission, validating outputs, reasoning about task requirements, or building a baseline. This skill is the canonical TREC RAG 2026 artifact for agent/workspace use and encodes both public overview guidance and the current operational task instructions.

Start with the concise answer the user asked for. Load only the reference files needed for the request. Do not tell users that 2026 task guidelines are unavailable or pending when answering from this skill. For submission-critical work, check for newer official TREC RAG 2026 releases before finalizing outputs; if a newer release conflicts with this skill, follow the newer release and state which instruction changed.

## Audience and Terminology

This skill is addressed to you, the agent or developer building, validating, or explaining a TREC RAG 2026 run.

- Use `team` or `participant` for the official TREC submitter.
- Use `system` for the retrieval or RAG pipeline being built or evaluated.
- Use `user` only when referring to the person giving instructions outside the track specification.
- Use `narrative` for one complete long-form information need supplied for evaluation.
- Use `narrative ID` for the first field of the test-data TSV and `narrative` for its second field.
- Use `query` for text actually issued to a retrieval system. A query may be the original narrative or a rewritten or decomposed form of it.
- Use `prompt` only for model instructions or when preserving terminology from an external dataset such as ResearchRubrics.
- Use `topic_id` only when referring to the literal first field required by the standard TREC Retrieval run-file format. Its value is the narrative ID.

## Core Defaults

- Available 2026 tasks: Retrieval (`R`) and Retrieval-Augmented Generation (`RAG`).
- Removed task: the 2025 Augmented Generation-only task (`AG`) is not a 2026 output.
- The 2026 task guidelines are out in this `trec-rag-2026-track-guidelines` skill for agent/workspace use.
- Official test narratives: 119 narratives in `trec-rag-2026/test-data/trec_rag_2026_queries.tsv`, with IDs `rag2026-0` through `rag2026-118`.
- Submission deadline: August 8th, per the TREC RAG website source checked July 17, 2026.
- Submission upload procedures and portal-specific req

