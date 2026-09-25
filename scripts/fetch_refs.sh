#!/usr/bin/env bash
# fetch_refs.sh — download authoritative bibtex/metadata for all cite keys
set -uo pipefail

OUT=refcheck
mkdir -p "$OUT/raw"
LOGFILE="$OUT/fetch.log"
: > "$LOGFILE"

UA="refcheck/1.0 (mailto:kong276818@gmail.com)"
MAX_RETRIES=3

fetch_with_retry() {
    local key="$1" url="$2" outfile="$3"
    local attempt code
    for attempt in 1 2 3; do
        code=$(curl -sS -L --max-time 30 -A "$UA" \
               -w '%{http_code}' \
               "$url" -o "$outfile" 2>/dev/null)
        if [[ "$code" == "200" ]]; then
            echo -e "${key}\t${code}" >> "$LOGFILE"
            return 0
        elif [[ "$code" == "404" ]]; then
            echo -e "${key}\t404" >> "$LOGFILE"
            echo "NOT_FOUND (HTTP 404)" > "$outfile"
            return 0
        else
            echo "  attempt $attempt failed: HTTP $code" >&2
            sleep 10
        fi
    done
    echo -e "${key}\tFETCH_FAILED_${code}" >> "$LOGFILE"
    echo "FETCH_FAILED (HTTP $code after $MAX_RETRIES attempts)" > "$outfile"
    return 1
}

total=0; skipped=0; ok=0; failed=0

while IFS=$'\t' read -r key type id; do
    [[ -z "${key:-}" || "${key:0:1}" == "#" ]] && continue
    ((total++))

    case "$type" in
        skip)
            echo -e "${key}\tSKIP\t${id}" >> "$LOGFILE"
            echo "SKIP: ${id}" > "$OUT/raw/${key}.txt"
            ((skipped++))
            continue
            ;;
        arxiv)
            url="https://arxiv.org/bibtex/${id}"
            ;;
        doi)
            url="https://api.crossref.org/works/${id}"
            ;;
        acl)
            url="https://aclanthology.org/${id}.bib"
            ;;
        *)
            echo -e "${key}\tBADTYPE\t${id}" >> "$LOGFILE"
            echo "BADTYPE: ${type}" > "$OUT/raw/${key}.txt"
            continue
            ;;
    esac

    echo -n "Fetching ${key} (${type}:${id}) ... "
    outfile="$OUT/raw/${key}.txt"

    if fetch_with_retry "$key" "$url" "$outfile"; then
        echo "OK"
        ((ok++))
    else
        echo "FAILED"
        ((failed++))
    fi

    # Rate-limit: arxiv asks for 3s between requests
    sleep 3

done < refs.tsv

echo ""
echo "Done: ${total} total, ${ok} OK, ${skipped} skipped, ${failed} failed"
echo "Log: ${LOGFILE}"
