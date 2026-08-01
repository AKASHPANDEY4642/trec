Title: Live Content

Description: Fetched live

Source: https://raw.githubusercontent.com/TREC-RAG/trec-rag-skills/main/skills/pyserini-rest-api/SKILL.md

---

---
name: pyserini-rest-api
description: Use for accessing the Pyserini REST API, which is the official API for the TREC RAG tracks.
metadata:
  version: v0.3.0
  source_url: https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/pyserini-rest-api
---

# Pyserini REST API

Use this skill when you need to access the Pyserini REST API or help someone build against it. Search is one route family exposed by the API, not the full API surface.

## Service Location

The Pyserini REST API is currently exposed at:

```text
http://api.castorini.uwaterloo.ca
```

The service location is liable to change. Consult the `pyserini-rest-api` skill in the https://github.com/TREC-RAG/trec-rag-skills/ repository for the latest service location and usage guidance.

Command examples use `<base-url>` as a placeholder for the current service location.

## Dataset Configuration

Use these exact dataset-to-index mappings:

- MS MARCO V2.1 Segmented Doc: `msmarco-v2.1-doc-segmented`
- ClimbMix: `climbmix-400b`
- FineWeb-Edu: `fineweb-edu-100b-karpathy`

When the user asks for a dataset by name, map it to the corresponding index above. If the user provides an explicit index, use it as given after confirming it matches the intended dataset when the context is ambiguous.

If the dataset or index is not clear from context, ask the user which index to search before making authenticated search or document-fetch requests. If the user asks which indexes are available, provide the dataset configuration above.

## Authentication Workflow

The Pyserini REST API requires a Pyserini access token. Use the repo-local workflow below unless the user has already provided another secure token mechanism. Token safety rules are mandatory.

### Token Access

If the user does not have a Pyserini API token, tell them to email `get-pyserini@googlegroups.com` to request one.

Mandatory token safety rules:

- Do not attempt authenticated searches or document fetches unless a token is available through a safe local mechanism.
- Never commit `.env.local`, never paste the token into chat, and never print it in command output.
- Never commit `.curlrc.pyserini-rest`, and never print its contents.
- Do not put the token in tracked files, examples, logs, shell history snippets, command lines, or skill documentation.

Recommended repo-local workflow:

- Ask the user for the Pyserini API token if it is not already available locally.
- Store the token in the repo-local `.env.local` file as `PYSERINI_API_TOKEN=...`.
- Prefer storing the curl authorization header in the repo-local `.curlrc.pyserini-rest` file.
- If `.env.local` already exists, read only enough to determine whether `PYSERINI_API_TOKEN` is present; do not display the file contents.
- If `.curlrc.pyserini-rest` is missing but `.env.local` has `PYSERINI_API_TOKEN`, create `.curlrc.pyserini-rest` with mode `600` and a single authorization header derived from the token.
- If `.curlrc.pyserini-rest` exists but authenticated requests fail after confirming `PYSERINI_API_TOKEN` is present, regenerate `.curlrc.pyserini-rest` from `.env.local` without printing either file.
- Use `.curlrc.pyserini-rest` for requests:

```bash
curl -sS -K .curlrc.pyserini-rest -o tmp/pyserini-rest-search.json "<base-url>/v1/climbmix-400b/search?query=anserini&hits=5"
jq . tmp/pyserini-rest-search.json
```

Rationale: using `curl -sS -K .curlrc.pyserini-rest` keeps the token out of visible command lines and creates a stable command prefix that can be approved once for network access. After that approval is persisted, future Pyserini REST requests can reuse the same prefix without repeated escalation prompts.

When using `jq`, prefer saving the `curl` response to a temporary JSON file with `-o` and then running `jq` as a separate local command. Do not pipe `curl` directly into `jq`; the sandbox treats each pipeline segment as a separate command and may require repeated escalation for otherwise local JSON inspection.

If the API returns an authorization error, tell the user the local token appears missing, expired, or invalid without revealing any token value.

## Endpoints

The service presents an OpenAPI-compliant REST API. Use the interactive and machine-readable documentation when discovering endpoints or generating clients:

- Swagger UI: `<base-url>/docs`
- ReDoc: `<base-url>/redoc`
- OpenAPI JSON: `<base-url>/openapi.json`
- OpenAPI YAML: `<base-url>/openapi.yaml`

Endpoint paths are relative to `<base-url>`:

- `GET /`
- `GET /v1/{index}/search`
- `GET /v1/{index}/doc/{docid}`

## Health Check

Use this procedure when the user asks whether the Pyserini REST API server is up.

Start with the unauthenticated root endpoint. This confirms that the HTTP service is reachable without needing to touch the local token:

```bash
curl -sS

