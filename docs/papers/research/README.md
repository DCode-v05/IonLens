# Quantifying Ion Energy Flux Across Plasma Sheaths — Research Monograph

Team Astrophage · DIA Techathon 2.0 (Naari Shakti) · Track: Plasma Technologies for Industry & Society

This folder contains a complete, Overleaf-ready LaTeX research monograph plus the
supporting research artifacts (reference list, BibTeX database, and literature
dossier). Everything is strictly black-and-white — no colour themes.

## What is in this folder

| File / folder | Purpose |
|---|---|
| `main.tex` | Master document. Compile this. |
| `frontmatter/` | Title page, abstract, declaration, acknowledgements, nomenclature. |
| `chapters/` | The 25 chapters (Parts I–V). |
| `appendices/` | Appendices A–E (derivations, notation, specs, data, annotated bibliography). |
| `refs/references_part1..4.bib` | The full BibTeX database of real references. |
| `refs/citation_index.json` | Machine-readable index (citekey ↔ title/topic). |
| `Research_Reference_List.md` | Human-readable list of every paper: title + authors + link, grouped by theme. |
| `Research_Dossier.tex` | Standalone literature-review dossier (compile separately). |

## How to compile (Overleaf — recommended, no install)

1. Go to https://www.overleaf.com and create a **New Project → Upload Project**.
2. Zip this `Research_Paper` folder and upload it (or drag the files in).
3. In Overleaf: **Menu → Settings → Compiler = pdfLaTeX**, **TeX Live version = latest**.
4. Bibliography backend is **Biber** (Overleaf detects this automatically from `biblatex`).
5. Click **Recompile**. The first build runs pdflatex → biber → pdflatex twice; give it a minute.
6. Compile `Research_Dossier.tex` as its own document for the standalone dossier.

## How to compile (local, if you install a TeX distribution)

Install MiKTeX (Windows) or TeX Live, then from this folder:

```
latexmk -pdf main.tex
```

`latexmk` runs pdflatex and biber the correct number of times automatically. If you
do not have `latexmk`, run manually:

```
pdflatex main
biber main
pdflatex main
pdflatex main
```

## Notes on integrity and scope

- Every reference was located through live search of public scientific databases
  (arXiv, Crossref, web search). Verify any citation you intend to quote directly.
- Physical numbers described as "typical"/"representative" are order-of-magnitude
  teaching values, not original measurements.
- The proposed diagnostic framework is a research proposal; performance statements
  are objectives to be validated by the experimental programme in Part IV.
- Final page count depends on the compiled formatting; the content is structured
  as a comprehensive monograph with a large bibliography and appendices.
