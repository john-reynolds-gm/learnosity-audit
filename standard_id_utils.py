"""
standard_id_utils.py

Shared standard-ID normalization helpers used by both build_database.py
(when building the crosswalk / reference join) and analyze_id_mismatches.py
(when diagnosing why a state's IDs don't match the crosswalk).

Previously build_database.py owned the real logic (isolate_standard_id /
normalize_separators) while analyze_id_mismatches.py reimplemented a
simpler depth-based prefix strip of its own. The two drifted: the audit
script could flag a "prefix issue" for a state that isolate_standard_id()
already handled, purely because the scripts didn't share code. Centralizing
here keeps both callers testing IDs the same way.

Requirements:
    pip install pandas
"""

import re

import pandas as pd


# A grade token is the anchor where a state's own standard id begins: a single
# grade (K, 1-8), a grade band (9-12, 9-10, 11-12), the 'HS' band marker, or
# a course-name grade like 'A1' (Algebra 1) -- mirrors the A1->9 handling in
# parse_alignment_guide.py's _grade_from_str(). Without 'A1' here, Algebra 1
# entries (e.g. '9.12.A1.A.APR.A.1') have no recognized grade token, fall
# through isolate_standard_id()'s "no grade token" branch, and never resolve
# to a state in build_state_standards() -- silently dropping them from the
# crosswalk under a null state.
# Restricted to 1-2 digit grades so 4-digit year segments (e.g. 'IN.2023...')
# are not mistaken for grades.
GRADE_TOKEN = re.compile(r'^(K|A1|\d{1,2}|\d{1,2}-\d{1,2}|HS)$')


def strip_hs_prefix(code: str) -> str:
    """Remove a leading 'HS' from a CCSS code, e.g. 'HSA.APR.A.1' -> 'A.APR.A.1'."""
    if pd.isna(code):
        return code
    return re.sub(r'^HS', '', code)


def normalize_separators(code: str) -> str:
    """Canonicalize the domain-cluster separator: dashes -> dots (A-APR -> A.APR)."""
    return code.replace('-', '.') if isinstance(code, str) else code


def isolate_standard_id(new_tag_value: str) -> str:
    """
    Strip the leading state/framework/subject metadata from a Learnosity
    newTagValue, returning the state-local standard id starting at the first
    grade token. 'SC.CCRS.MA.6.GM.3' -> '6.GM.3'; 'OH.Math.1.G' -> '1.G'.
    Falls back to the whole value when no grade token is present (HS-only codes).

    A grade token match is only accepted if at least one segment follows it
    (domain/cluster/standard). Without this check, a high-school standard
    with no numeric grade prefix -- e.g. 'NJ.SLS.MA.A-APR.A.1', where the
    trailing '.1' is the standard's own number, not a grade -- gets its
    last segment mistaken for a grade token, collapsing many distinct HS
    standards (A-APR.A.1, A-CED.A.1, F-IF.A.1, ...) down to the same bare
    '1'. Requiring a following segment excludes that false positive while
    still matching genuine cases like '...1.G.3' (grade token '1' followed
    by 'G.3').

    Some states (observed: NY) glue their abbreviation directly onto the
    grade with a dash instead of a dot within the same dot-segment, e.g.
    'MA.NY-1.MD.3a'. Splitting on '.' alone leaves 'NY-1' as one token,
    which matches no grade pattern, so the whole value falls through to
    the unstripped fallback. This is handled as a second check per segment:
    if the segment contains a dash, split on it and check whether the part
    *after* the dash is itself a grade token -- but only when the part
    *before* the dash is non-numeric, so a genuine grade band like '9-12'
    (numeric on both sides) is never misinterpreted this way.
    """
    if not isinstance(new_tag_value, str):
        return new_tag_value
    parts = new_tag_value.split('.')
    for i, p in enumerate(parts):
        if GRADE_TOKEN.match(p) and i < len(parts) - 1:
            return '.'.join(parts[i:])
        if '-' in p:
            pre, _, post = p.partition('-')
            if (not re.match(r'^\d+$', pre) and GRADE_TOKEN.match(post)
                    and i < len(parts) - 1):
                return '.'.join([post] + parts[i + 1:])
    return new_tag_value  # no grade token (e.g. 'MA.AR.A-APR') — best-effort


def add_subpart_dot(standard_id: str) -> str:
    """
    Insert a '.' before a trailing lowercase sub-part letter that directly
    follows a digit, e.g. '3.M.B.3a' -> '3.M.B.3.a', to match the crosswalk's
    convention. Leaves an already-dotted id ('3.M.B.3.a') and anything not
    matching this exact shape untouched.

    Some reference-file IDs (verified: NJ) omit the period before a trailing
    lowercase sub-part letter. This is applied only as a fallback -- callers
    should prefer the un-modified id and use the dotted form solely when the
    plain form doesn't already match, so a correct id is never rewritten.
    """
    if not isinstance(standard_id, str):
        return standard_id
    return re.sub(r'(\d)([a-z])$', r'\1.\2', standard_id)
