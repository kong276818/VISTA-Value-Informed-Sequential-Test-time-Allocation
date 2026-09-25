#!/usr/bin/env python3
"""
compare_refs.py — compare fetched metadata against expected and bbl.stale.

Inputs:
  refcheck/raw/*.txt   — fetched bibtex / crossref JSON / ACL bibtex
  paper.bbl.stale      — locally compiled bibliography (22 entries, old keys)

Output:
  refcheck/report.tsv  — key, status, verdict, mismatch_fields, ...
  refcheck/clean_bibtex.bib — assembled bibtex entries with paper.tex cite keys
"""

import re, json, os, sys
from pathlib import Path

RAW_DIR   = Path("refcheck/raw")
REPORT    = Path("refcheck/report.tsv")
CLEAN_BIB = Path("refcheck/clean_bibtex.bib")
BBL_FILE  = Path("paper.bbl.stale")

# ---------------------------------------------------------------------------
# Expected metadata: (title_fragment_lower, first_author_last_partial, year_lo, year_hi)
# title_fragment: substring that should appear in normalized fetched title
# first_author_last_partial: prefix/substring of first author last name
# ---------------------------------------------------------------------------
EXPECTED = {
    'akgul2025lynx':          ('lynx',                    'akgül',        2025, 2025),
    'angelopoulos2024crc':    ('conformal risk control',  'angelopoulos', 2022, 2025),
    'angelopoulos2025ltt':    ('learn then test',         'angelopoulos', 2021, 2025),
    'banino2021pondernet':    ('pondernet',                'banino',       2021, 2021),
    'bates2021rcps':          ('risk-controlling',         'bates',        2021, 2021),
    'brown2024monkeys':       ('monkeys',                  'brown',        2024, 2024),
    'chen2025overthinking':   ('overthinking',             'chen',         2024, 2025),
    'cheshmi2025earlyrejection':('early rejection',        'cheshmi',      2025, 2025),
    'clark2018arc':           ('arc',                      'clark',        2018, 2018),
    'cobbe2021verifiers':     ('verifiers',                'cobbe',        2021, 2021),
    'deepseekr1':             ('deepseek-r1',              'deepseek',     2025, 2026),
    'gneiting2007strictly':   ('strictly proper',          'gneiting',     2007, 2007),
    'graves2016adaptive':     ('adaptive computation',     'graves',       2016, 2016),
    'guo2017calibration':     ('calibration of modern',   'guo',          2017, 2017),
    'horvitz1952sampling':    ('sampling without replacement','horvitz',   1952, 1952),
    'howard2021confidence':   ('confidence sequences',     'howard',       2018, 2022),
    'jin2021medqa':           ('disease',                  'jin',          2021, 2021),
    'kadavath2022know':       ('know what they know',      'kadavath',     2022, 2022),
    'kojima2022zeroshot':     ('zero-shot',                'kojima',       2022, 2023),
    'kuhn2023semantic':       ('semantic uncertainty',     'kuhn',         2023, 2023),
    'lei2018conformal':       ('distribution-free',        'lei',          2017, 2018),
    'liu2020fastbert':        ('fastbert',                 'liu',          2020, 2020),
    'liu2025answerconvergence':('answer convergence',      'liu',          2025, 2025),
    'manakul2023selfcheckgpt':('selfcheckgpt',             'manakul',      2023, 2023),
    'nagle2026terminator':    ('terminator',               'nagle',        2026, 2026),
    'pal2022medmcqa':         ('medmcqa',                  'pal',          2022, 2022),
    'quach2024conformal':     ('conformal language',       'quach',        2023, 2024),
    'qwen3':                  ('qwen3',                    '',             2025, 2025),
    'ramdas2023safe':         ('game-theoretic',           'ramdas',       2022, 2023),
    'schwartz2020righttool':  ('right tool',               'schwartz',     2020, 2020),
    'snell2025scaling':       ('scaling',                  'snell',        2024, 2025),
    'sun2026refrain':         ('stop when enough',         'sun',          2025, 2026),
    'uesato2022process':      ('process',                  'uesato',       2022, 2022),
    'wang2023selfconsistency':('self-consistency',         'wang',         2022, 2023),
    'wang2024mmlu':           ('mmlu',                     'wang',         2024, 2024),
    'wang2026conformalthinking':('conformal thinking',     'wang',         2026, 2026),
    'waudbysmith2024betting': ('betting',                  'waudby',       2022, 2024),
    'wei2022cot':             ('chain-of-thought',         'wei',          2022, 2023),
    'wu2025thoughtcalibration':('thought calibration',     'wu',           2025, 2025),
    'xie2026statistical':     ('statistical early stopping','xie',         2025, 2026),
    'xin2020deebert':         ('deebert',                  'xin',          2020, 2020),
    'xiong2024uncertainty':   ('uncertainty',              'xiong',        2023, 2024),
    'yang2026deer':           ('dynamic early exit',       'yang',         2025, 2026),
    'yao2023tree':            ('tree of thoughts',         'yao',          2023, 2023),
    'zeng2025thinkingoutloud':('thinking out loud',        'zeng',         2025, 2025),
    'zhai2026adaptive':       ('adaptive test-time',       'zhai',         2026, 2026),
    'zhang2025selfverification':('self-verification',      'zhang',        2025, 2025),
    'zhou2026overthinking':   ('overthinking',             'zhou',         2026, 2026),
    'yu2026bpac':             ('anytime safe',             'yu',           2026, 2026),
    # skip entries — no fetch
    'brier1950verification':  None,
    'murphy1973new':          None,
    'vovk2005algorithmic':    None,
}

# Map paper.tex key → bbl.stale key (for the 22 papers that exist in bbl.stale)
KEY_TO_BBL = {
    'snell2025scaling':       'snell2024scaling',
    'angelopoulos2025ltt':    'angelopoulos2021learn',
    'angelopoulos2024crc':    'angelopoulos2024conformal',
    'howard2021confidence':   'howard2021time',
    'ramdas2023safe':         'ramdas2023game',
    'wei2022cot':             'wei2022chain',
    # same key in both
    'qwen3':                  'qwen3',
    'deepseekr1':             'deepseekr1',
    'banino2021pondernet':    'banino2021pondernet',
    'graves2016adaptive':     'graves2016adaptive',
    'quach2024conformal':     'quach2024conformal',
    'wang2024mmlu':           'wang2024mmlu',
    'clark2018arc':           'clark2018arc',
    'pal2022medmcqa':         'pal2022medmcqa',
    'jin2021medqa':           'jin2021medqa',
    'brier1950verification':  'brier1950verification',
    'gneiting2007strictly':   'gneiting2007strictly',
    'murphy1973new':          'murphy1973new',
}

# -------------------------------------------------------------------------
# Parsers
# -------------------------------------------------------------------------

def normalize(s):
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    s = s.lower()
    s = re.sub(r'[{}\\]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_bibtex_fields(text):
    """Extract title, author, year from a @misc/@inproceedings bibtex entry."""
    out = {}
    # Remove the outer @type{key, ... }
    m = re.search(r'@\w+\s*\{[^,]+,(.+)\}\s*$', text, re.S)
    body = m.group(1) if m else text

    def extract_field(field):
        # Match  field = {value}  or  field = "value"  (handles nested braces)
        pat = re.compile(
            r'\b' + re.escape(field) + r'\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            re.I | re.S
        )
        m2 = pat.search(body)
        if m2:
            return m2.group(1).strip()
        pat2 = re.compile(
            r'\b' + re.escape(field) + r'\s*=\s*"([^"]*)"', re.I | re.S
        )
        m3 = pat2.search(body)
        if m3:
            return m3.group(1).strip()
        # bare number for year
        if field == 'year':
            m4 = re.search(r'\byear\s*=\s*(\d{4})', body, re.I)
            if m4:
                return m4.group(1)
        return ''

    out['title']  = extract_field('title')
    out['author'] = extract_field('author')
    out['year']   = extract_field('year')
    # eprint / doi
    out['eprint'] = extract_field('eprint')
    out['doi']    = extract_field('doi')
    return out

def parse_crossref_json(text):
    """Extract title, author, year from Crossref API JSON."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    msg = data.get('message', data)
    titles = msg.get('title', [])
    title = titles[0] if titles else ''
    authors_raw = msg.get('author', [])
    authors = ' and '.join(
        (a.get('family', '') + ', ' + a.get('given', '')).strip(', ')
        for a in authors_raw
    )
    year = ''
    dp = msg.get('published', msg.get('published-print', msg.get('published-online', {})))
    parts = dp.get('date-parts', [[]])
    if parts and parts[0]:
        year = str(parts[0][0])
    doi = msg.get('DOI', '')
    return {'title': title, 'author': authors, 'year': year, 'eprint': '', 'doi': doi}

def parse_bbl(bbl_text):
    """Parse a .bbl file into dict: key → {title, author, year, venue}."""
    entries = {}
    # Split on \bibitem
    chunks = re.split(r'\\bibitem', bbl_text)
    for chunk in chunks[1:]:
        # Extract key: \bibitem[...]{key}
        km = re.match(r'\[.*?\]\{([^}]+)\}', chunk)
        if not km:
            km = re.match(r'\{([^}]+)\}', chunk)
        if not km:
            continue
        key = km.group(1)
        # Clean up the text block
        body = chunk[km.end():]
        # Authors: first \newblock-delimited section (or first non-blank line after key)
        lines = [l.strip() for l in body.split('\n') if l.strip()]
        author = ''
        title = ''
        year = ''
        # First non-empty content = author line
        if lines:
            author = re.sub(r'\\newblock\b', '', lines[0]).strip().rstrip('.')
        # \newblock Title.
        nb_parts = re.split(r'\\newblock', body)
        if len(nb_parts) >= 2:
            title = nb_parts[1].strip().rstrip('.')
        if len(nb_parts) >= 3:
            venue_line = nb_parts[2].strip()
        # Year: last 4-digit number in the entry
        ym = re.findall(r'\b(1[89]\d\d|20[0-9]{2})\b', body)
        if ym:
            year = ym[-1]
        entries[key] = {'author': author, 'title': title, 'year': year}
    return entries

# -------------------------------------------------------------------------
# Load bbl.stale
# -------------------------------------------------------------------------
bbl_data = {}
if BBL_FILE.exists():
    bbl_data = parse_bbl(BBL_FILE.read_text())

# -------------------------------------------------------------------------
# Read refs.tsv for key→type mapping
# -------------------------------------------------------------------------
refs = {}
with open('refs.tsv') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) >= 3:
            refs[parts[0]] = (parts[1], parts[2])

# -------------------------------------------------------------------------
# Process each key
# -------------------------------------------------------------------------
rows = []
clean_bibs = []

for key in sorted(EXPECTED.keys()):
    exp = EXPECTED[key]
    ref_type, ref_id = refs.get(key, ('skip', ''))

    raw_file = RAW_DIR / f"{key}.txt"

    # ---- determine status ----
    if ref_type == 'skip' or not raw_file.exists():
        rows.append({
            'key': key, 'type': ref_type, 'id': ref_id,
            'status': 'SKIP', 'verdict': 'MANUAL',
            'fetched_title': '', 'fetched_authors': '', 'fetched_year': '',
            'bbl_title': '', 'bbl_authors': '', 'bbl_year': '',
            'mismatch_fields': 'manual citation required',
            'clean_bib': '',
        })
        continue

    raw_text = raw_file.read_text(encoding='utf-8', errors='replace').strip()

    if raw_text.startswith('SKIP:'):
        rows.append({
            'key': key, 'type': ref_type, 'id': ref_id,
            'status': 'SKIP', 'verdict': 'MANUAL',
            'fetched_title': '', 'fetched_authors': '', 'fetched_year': '',
            'bbl_title': '', 'bbl_authors': '', 'bbl_year': '',
            'mismatch_fields': 'manual citation required', 'clean_bib': '',
        })
        continue

    if 'NOT_FOUND' in raw_text:
        rows.append({
            'key': key, 'type': ref_type, 'id': ref_id,
            'status': 'NOT_FOUND', 'verdict': 'NOT_FOUND',
            'fetched_title': '', 'fetched_authors': '', 'fetched_year': '',
            'bbl_title': '', 'bbl_authors': '', 'bbl_year': '',
            'mismatch_fields': f'HTTP 404 for {ref_type}:{ref_id}', 'clean_bib': '',
        })
        continue

    if 'FETCH_FAILED' in raw_text:
        rows.append({
            'key': key, 'type': ref_type, 'id': ref_id,
            'status': 'FETCH_FAILED', 'verdict': 'FETCH_FAILED',
            'fetched_title': '', 'fetched_authors': '', 'fetched_year': '',
            'bbl_title': '', 'bbl_authors': '', 'bbl_year': '',
            'mismatch_fields': raw_text, 'clean_bib': '',
        })
        continue

    # ---- parse fetched content ----
    if ref_type == 'doi':
        fields = parse_crossref_json(raw_text)
    else:
        fields = parse_bibtex_fields(raw_text)

    f_title   = fields.get('title', '')
    f_authors = fields.get('author', '')
    f_year    = fields.get('year', '')
    f_eprint  = fields.get('eprint', '')
    f_doi     = fields.get('doi', '')

    # ---- bbl.stale comparison ----
    bbl_key = KEY_TO_BBL.get(key)
    bbl_entry = bbl_data.get(bbl_key, {}) if bbl_key else {}
    bbl_title   = bbl_entry.get('title', '')
    bbl_authors = bbl_entry.get('author', '')
    bbl_year    = bbl_entry.get('year', '')

    # ---- verdict logic ----
    mismatches = []

    if exp is None:
        verdict = 'MANUAL'
        rows.append({
            'key': key, 'type': ref_type, 'id': ref_id,
            'status': 'OK', 'verdict': verdict,
            'fetched_title': f_title, 'fetched_authors': f_authors, 'fetched_year': f_year,
            'bbl_title': bbl_title, 'bbl_authors': bbl_authors, 'bbl_year': bbl_year,
            'mismatch_fields': '', 'clean_bib': raw_text,
        })
        continue

    exp_title_frag, exp_first_auth_lower, yr_lo, yr_hi = exp

    # Title check
    norm_title = normalize(f_title)
    if exp_title_frag and exp_title_frag not in norm_title:
        mismatches.append(f'title(expected fragment "{exp_title_frag}" not in "{norm_title[:80]}")')

    # Year check
    try:
        yr_int = int(f_year)
        if not (yr_lo <= yr_int <= yr_hi):
            mismatches.append(f'year(fetched={f_year}, expected {yr_lo}-{yr_hi})')
    except (ValueError, TypeError):
        if f_year:
            mismatches.append(f'year(unparseable: {f_year!r})')

    # First-author check (compare lowercase last name fragments)
    if exp_first_auth_lower:
        # Get first author last name
        auth_parts = re.split(r'\s+and\s+', f_authors, flags=re.I)
        first_auth = auth_parts[0] if auth_parts else ''
        # Normalize: remove braces, get last name portion
        first_auth_norm = normalize(first_auth.split(',')[0])
        # Also try "Firstname Lastname" format
        if not first_auth_norm:
            first_auth_norm = normalize(first_auth)
        # Check unicode match too (ü ↔ ü)
        fa_ascii = first_auth_norm.encode('ascii', 'ignore').decode()
        exp_ascii = exp_first_auth_lower.encode('ascii', 'ignore').decode()
        if exp_first_auth_lower not in first_auth_norm and exp_ascii not in fa_ascii:
            mismatches.append(f'first_author(fetched="{first_auth_norm[:40]}", expected contains "{exp_first_auth_lower}")')

    # bbl.stale author cross-check (if we have bbl data for this key)
    if bbl_entry and bbl_authors:
        # Check if bbl claimed different authors from what we fetched
        norm_fetched_auth = normalize(f_authors)
        norm_bbl_auth = normalize(bbl_authors)
        # Extract last names from both
        def lastnames(s):
            parts = re.split(r'\s+and\s+|,\s+(?=[A-Z])', s, flags=re.I)
            return set(normalize(p.split(',')[0]) for p in parts if p.strip())
        f_lns = lastnames(f_authors)
        b_lns = lastnames(bbl_authors)
        if f_lns and b_lns and f_lns != b_lns:
            only_in_bbl    = b_lns - f_lns
            only_in_fetch  = f_lns - b_lns
            if only_in_bbl or only_in_fetch:
                mismatches.append(
                    f'authors_vs_bbl(bbl_only={sorted(only_in_bbl)}, fetch_only={sorted(only_in_fetch)})'
                )

    if mismatches:
        verdict = 'MISMATCH'
    else:
        verdict = 'PASS'

    # ---- build clean bibtex with paper.tex cite key ----
    # Rewrite the arXiv auto-key to our paper.tex key
    clean_bib = re.sub(r'(@\w+\s*\{)[^,]+,', r'\g<1>' + key + ',', raw_text, count=1)
    if verdict == 'PASS' or (verdict == 'MISMATCH' and 'title' not in ' '.join(mismatches)):
        clean_bibs.append(clean_bib)

    rows.append({
        'key': key, 'type': ref_type, 'id': ref_id,
        'status': 'OK', 'verdict': verdict,
        'fetched_title': f_title[:100], 'fetched_authors': f_authors[:120], 'fetched_year': f_year,
        'bbl_title': bbl_title[:80], 'bbl_authors': bbl_authors[:80], 'bbl_year': bbl_year,
        'mismatch_fields': ' | '.join(mismatches), 'clean_bib': clean_bib,
    })

# -------------------------------------------------------------------------
# Write report.tsv
# -------------------------------------------------------------------------
REPORT.parent.mkdir(parents=True, exist_ok=True)
header = ['key','type','id','status','verdict','mismatch_fields',
          'fetched_title','fetched_year','fetched_authors',
          'bbl_title','bbl_year','bbl_authors']
with open(REPORT, 'w') as f:
    f.write('\t'.join(header) + '\n')
    for r in rows:
        f.write('\t'.join(str(r.get(h, '')) for h in header) + '\n')

# -------------------------------------------------------------------------
# Write clean_bibtex.bib
# -------------------------------------------------------------------------
with open(CLEAN_BIB, 'w') as f:
    f.write("% Auto-assembled from fetch_refs.sh + compare_refs.py\n")
    f.write("% Entries with verdict=PASS or non-title MISMATCH only\n\n")
    for bib in clean_bibs:
        f.write(bib.strip() + '\n\n')

# -------------------------------------------------------------------------
# Print summary table
# -------------------------------------------------------------------------
verdicts = {}
for r in rows:
    v = r['verdict']
    verdicts[v] = verdicts.get(v, 0) + 1

print(f"\n{'='*70}")
print(f"{'KEY':<35} {'TYPE':<7} {'VERDICT':<14} {'MISMATCH / NOTE'}")
print(f"{'='*70}")

for r in sorted(rows, key=lambda x: (x['verdict'], x['key'])):
    flag = '*** ' if r['verdict'] not in ('PASS', 'MANUAL', 'SKIP') else '    '
    note = r['mismatch_fields'][:55] if r['mismatch_fields'] else ''
    print(f"{flag}{r['key']:<33} {r['type']:<7} {r['verdict']:<14} {note}")

print(f"\nSummary:")
for v, n in sorted(verdicts.items()):
    print(f"  {v}: {n}")
print(f"\nReport: {REPORT}")
print(f"Clean bibtex: {CLEAN_BIB}")
