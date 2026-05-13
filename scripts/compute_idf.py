#!/usr/bin/env python3
"""
Compute IDF (inverse document frequency) for all terms across all
conversations, per user. Store as a JSON blob in Postgres for the
predictions edge function to read.

IDF gives rare terms more weight than common ones. Currently the BoW
cosine treats "really" and "EthicsPoint" as equal — TF-IDF fixes that.

Run this once per user when the corpus has grown enough to be useful.
Re-run periodically as new conversations come in.

Usage:
    python3 -m scripts.compute_idf            # current user
    python3 -m scripts.compute_idf --top-k 5000
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ndp.config import load_config

# Match the edge function's stopword set
STOP = set("""
a about above after again against all am an and any are aren as at be because been before being below
between both but by can couldn did didn do does doesn doing don down during each few for from further
had hadn has hasn have haven having he her here hers herself him himself his how i if in into is isn it
its itself just me mightn more most mustn my myself needn no nor not now of off on once only or other
our ours ourselves out over own same shan she should shouldn so some such than that the their theirs
them themselves then there these they this those through to too under until up very was wasn we were
weren what when where which while who whom why will with won wouldn you your yours yourself yourselves
yeah okay really actually basically literally like gonna thing things stuff something someone anything
anyone everything everyone nothing never always sometimes often maybe perhaps think thought know knew
want wanted need needed got get goes went come came make made say said told tell asked way time day
year good bad big little old new
""".split())

WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def tokens(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOP}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5000,
                        help="Save IDF for the top-K most useful terms")
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()

    cfg = load_config()
    user_id = args.user_id or cfg["user_id"]

    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])

    print(f"Computing IDF for user_id={user_id}...")
    # Pull all conversation contents (paginate)
    contents = []
    offset = 0
    batch = 200
    while True:
        res = (client.table("ndp_conversations")
               .select("session_id, content")
               .eq("user_id", user_id)
               .range(offset, offset + batch - 1)
               .execute())
        if not res.data:
            break
        contents.extend(res.data)
        offset += batch
        print(f"  loaded {len(contents)} conversations...")

    n_docs = len(contents)
    if n_docs == 0:
        print("No conversations.")
        return

    print(f"\n{n_docs} conversations total. Computing document frequency...")
    df: Counter = Counter()
    for c in contents:
        for t in tokens(c.get("content", "")):
            df[t] += 1

    print(f"Vocabulary size: {len(df)} unique terms")

    # IDF = log(N / df), clamp to [0, log(N)]
    idf = {term: math.log(n_docs / count) for term, count in df.items()}

    # Keep only top-K most useful (highest IDF means rarer, more specific)
    sorted_terms = sorted(idf.items(), key=lambda x: x[1], reverse=True)
    top_idf = dict(sorted_terms[: args.top_k])

    print(f"Storing top {len(top_idf)} terms as IDF blob")
    print(f"  highest IDF: {sorted_terms[0]}")
    print(f"  lowest kept IDF: {sorted_terms[args.top_k - 1] if args.top_k <= len(sorted_terms) else sorted_terms[-1]}")

    # Store as a single row in a new ndp_idf table
    try:
        client.table("ndp_idf").upsert({
            "user_id": user_id,
            "n_documents": n_docs,
            "vocab_size": len(df),
            "idf": top_idf,
        }, on_conflict="user_id").execute()
        print(f"\nWrote IDF blob to ndp_idf for user {user_id}")
    except Exception as e:
        print(f"\nERROR: ndp_idf table may not exist yet. Run the migration first.")
        print(f"  detail: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
