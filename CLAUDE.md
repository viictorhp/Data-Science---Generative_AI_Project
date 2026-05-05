# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Early-stage research project on the textile and fashion industry in Spain. Currently consists of PDF documents and placeholder config files; no source code exists yet.

## Current State

- `.env` — empty, intended for environment variables (API keys, etc.)
- `requirements.txt` — empty, intended for Python dependencies
- `docs/` — five PDF source documents:
  - `informe-economico-de-la-moda-en-espana-2025.pdf` — economic report on Spanish fashion (2025)
  - `presentaciones_sectoriales_textil.pdf` — sector presentations on textiles
  - `memoria_anual_inditex_2025.pdf` — Inditex annual report (2025)
  - `circularidad_textil.pdf` — circular economy in textiles
  - `comercio_textil_2024.pdf` — textile trade data (2024)

## Development Setup

Once Python dependencies are added to `requirements.txt`:

```bash
pip install -r requirements.txt
```

Environment variables go in `.env` and should be loaded via `python-dotenv` or similar.
