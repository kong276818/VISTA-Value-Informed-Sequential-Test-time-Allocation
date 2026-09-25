#!/usr/bin/env python3
"""
build_bib.py — assemble authoritative mvtcs.bib from fetched raw files.

Entry-type policy:
  ACL anthology fetches  → keep @inproceedings as-is (rename key)
  DOI/Crossref fetches   → construct @article
  arXiv fetches          → @misc with known venue in note where applicable
  Manual entries         → hardcoded @article / @book
"""

import re, json
from pathlib import Path

RAW = Path("refcheck/raw")
OUT = Path("mvtcs.bib")

# ---------------------------------------------------------------------------
# Known published venues for arXiv papers (used in 'note' field)
# ---------------------------------------------------------------------------
KNOWN_VENUE = {
    'wei2022cot':           'NeurIPS 2022',
    'kojima2022zeroshot':   'NeurIPS 2022',
    'wang2023selfconsistency':'ICLR 2023',
    'yao2023tree':          'NeurIPS 2023',
    'kuhn2023semantic':     'ICLR 2023',
    'manakul2023selfcheckgpt':'EMNLP 2023',
    'guo2017calibration':   'ICML 2017',
    'quach2024conformal':   'ICLR 2024',
    'angelopoulos2024crc':  'ICLR 2024',
    'xiong2024uncertainty': 'ICLR 2024',
    'brown2024monkeys':     'ICLR 2025',
    'snell2025scaling':     'ICLR 2025',
    'wang2024mmlu':         'NeurIPS 2024',
    'yang2026deer':         'ICLR 2026',
    'xie2026statistical':   'ICML 2026',
    'cheshmi2025earlyrejection':'EMNLP 2025 Findings',
    'zeng2025thinkingoutloud':'EMNLP 2025',
    'nagle2026terminator':  'ICLR 2026',
    'zhang2025selfverification':'COLM 2025',
    'bates2021rcps':        'Journal of the ACM 2021',
}

# Year overrides (arXiv posting year differs from cite-key year / published year)
YEAR_OVERRIDE = {
    'graves2016adaptive':    '2016',  # arXiv says 2017 (last update); first posted 2016
    'jin2021medqa':          '2021',  # arXiv says 2020; Applied Sciences published 2021
    'waudbysmith2024betting':'2024',  # arXiv says 2022; JRSS-B published 2024
}

# Corporate-author papers: collapse individual author list to short form
CORPORATE_AUTHOR = {
    'deepseekr1': r'DeepSeek-AI',
    'qwen3':      r'Qwen Team',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bibtex(text: str) -> dict:
    """Return dict with keys: entry_type, bib_key, title, author, year,
    eprint, archivePrefix, primaryClass, doi, booktitle, pages, publisher, url."""
    out = {}
    m = re.match(r'@(\w+)\s*\{([^,]+),', text)
    if m:
        out['entry_type'] = m.group(1).lower()
        out['bib_key']    = m.group(2).strip()
    else:
        out['entry_type'] = 'misc'
        out['bib_key']    = ''

    def get(field):
        # {nested} value
        p = re.compile(
            r'\b' + re.escape(field) + r'\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            re.I | re.S
        )
        m2 = p.search(text)
        if m2:
            return m2.group(1).strip()
        # bare number
        if field == 'year':
            m3 = re.search(r'\byear\s*=\s*(\d{4})', text, re.I)
            if m3:
                return m3.group(1)
        return ''

    for f in ('title','author','year','eprint','archivePrefix','primaryClass',
              'doi','booktitle','pages','publisher','url','address','month','editor',
              'volume','number','note','ISBN'):
        out[f] = get(f)
    return out

def crossref_to_article(text: str, key: str) -> str:
    """Construct @article entry from Crossref JSON."""
    data = json.loads(text)
    m = data.get('message', data)
    title   = (m.get('title',['']) or [''])[0]
    authors_raw = m.get('author', [])
    def fmt_author(a):
        family = a.get('family','')
        given  = a.get('given','')
        if family and given:
            return f"{family}, {given}"
        return family or given
    author  = ' and '.join(fmt_author(a) for a in authors_raw)
    dp      = m.get('published', m.get('issued', {})).get('date-parts', [[]])
    year    = str(dp[0][0]) if dp and dp[0] else ''
    journal = (m.get('container-title',['']) or [''])[0]
    volume  = m.get('volume','')
    number  = m.get('issue','')
    pages   = m.get('page','')
    doi     = m.get('DOI','')
    return (
        f"@article{{{key},\n"
        f"  title   = {{{title}}},\n"
        f"  author  = {{{author}}},\n"
        f"  journal = {{{journal}}},\n"
        + (f"  volume  = {{{volume}}},\n" if volume else '')
        + (f"  number  = {{{number}}},\n" if number else '')
        + (f"  pages   = {{{pages}}},\n" if pages else '')
        + (f"  year    = {{{year}}},\n" if year else '')
        + (f"  doi     = {{{doi}}},\n" if doi else '')
        + f"}}"
    )

def arxiv_to_misc(fields: dict, key: str) -> str:
    """Construct @misc entry from parsed arXiv bibtex fields."""
    author  = CORPORATE_AUTHOR.get(key, fields.get('author',''))
    year    = YEAR_OVERRIDE.get(key, fields.get('year',''))
    eprint  = fields.get('eprint','')
    pclass  = fields.get('primaryClass','')
    title   = fields.get('title','')
    doi     = fields.get('doi','')
    note_venue = KNOWN_VENUE.get(key,'')
    note    = f"Published at {note_venue}" if note_venue else ''
    lines   = [f"@misc{{{key},"]
    lines.append(f"  title        = {{{title}}},")
    lines.append(f"  author       = {{{author}}},")
    lines.append(f"  year         = {{{year}}},")
    if eprint:
        lines.append(f"  eprint       = {{{eprint}}},")
        lines.append(f"  archivePrefix= {{arXiv}},")
    if pclass:
        lines.append(f"  primaryClass = {{{pclass}}},")
    if doi:
        lines.append(f"  doi          = {{{doi}}},")
    if note:
        lines.append(f"  note         = {{{note}}},")
    lines.append("}")
    return '\n'.join(lines)

def acl_to_inproceedings(raw_text: str, key: str) -> str:
    """Rename key in an ACL @inproceedings entry."""
    return re.sub(r'(@\w+\s*\{)[^,]+,', r'\g<1>' + key + ',', raw_text.strip(), count=1)

# ---------------------------------------------------------------------------
# Manual entries (no online source)
# ---------------------------------------------------------------------------
MANUAL_ENTRIES = {
    'brier1950verification': """\
@article{brier1950verification,
  title   = {Verification of Forecasts Expressed in Terms of Probability},
  author  = {Brier, Glenn W.},
  journal = {Monthly Weather Review},
  volume  = {78},
  number  = {1},
  pages   = {1--3},
  year    = {1950},
  doi     = {10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2},
}""",

    'murphy1973new': """\
@article{murphy1973new,
  title   = {A New Vector Partition of the Probability Score},
  author  = {Murphy, Allan H.},
  journal = {Journal of Applied Meteorology},
  volume  = {12},
  number  = {4},
  pages   = {595--600},
  year    = {1973},
  doi     = {10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2},
}""",

    'vovk2005algorithmic': """\
@book{vovk2005algorithmic,
  title     = {Algorithmic Learning in a Random World},
  author    = {Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn},
  publisher = {Springer},
  year      = {2005},
  isbn      = {978-0-387-00152-4},
}""",
}

# ---------------------------------------------------------------------------
# Entry-type decisions per cite key
# source: 'acl' | 'doi' | 'arxiv' | 'manual'
# ---------------------------------------------------------------------------
refs = {}
with open('refs.tsv') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) >= 3:
            refs[parts[0]] = (parts[1], parts[2])

# ---------------------------------------------------------------------------
# Build each entry
# ---------------------------------------------------------------------------
entries = []

for key, (src, ref_id) in sorted(refs.items()):
    raw_file = RAW / f"{key}.txt"

    if src == 'skip' or key in MANUAL_ENTRIES:
        entry = MANUAL_ENTRIES.get(key, f"% MANUAL NEEDED: {key}")
        entries.append((key, entry))
        continue

    if not raw_file.exists():
        entries.append((key, f"% MISSING RAW: {key}"))
        continue

    raw_text = raw_file.read_text(encoding='utf-8', errors='replace').strip()

    if 'NOT_FOUND' in raw_text or 'FETCH_FAILED' in raw_text:
        entries.append((key, f"% FETCH ERROR: {key} ({src}:{ref_id})"))
        continue

    if src == 'acl':
        entry = acl_to_inproceedings(raw_text, key)
    elif src == 'doi':
        entry = crossref_to_article(raw_text, key)
    else:  # arxiv
        fields = parse_bibtex(raw_text)
        entry  = arxiv_to_misc(fields, key)

    entries.append((key, entry))

# ---------------------------------------------------------------------------
# Write mvtcs.bib
# ---------------------------------------------------------------------------
bib_content = (
    "% mvtcs.bib — verified from primary sources (arXiv / Crossref / ACL Anthology)\n"
    "% Generated by build_bib.py; do not edit manually.\n"
    "% Keys match cite keys in paper.tex.\n\n"
)

for key, entry in entries:
    bib_content += entry.strip() + '\n\n'

OUT.write_text(bib_content, encoding='utf-8')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_ok      = sum(1 for k, e in entries if not e.startswith('%'))
n_missing = sum(1 for k, e in entries if e.startswith('%'))
print(f"Written: {OUT}  ({n_ok} entries, {n_missing} missing/errors)")
for k, e in entries:
    if e.startswith('%'):
        print(f"  PROBLEM: {e}")
