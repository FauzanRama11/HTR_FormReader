"""
terbilang.py
=============
Small, self-contained parser for Indonesian spelled-out numbers ("terbilang"
-- e.g. "lima juta rupiah", "sepuluh ribu") back into an integer, so a
digit value read from a form can be cross-checked against its own
spelled-out counterpart printed on the SAME form (nominal_penempatan and
reward_tunai's detail amount both have a genuine "Rp………… (…………)" digit +
parenthetical-words format -- see vlm.py's DIRECT_SEMANTIC_PROMPT/
REWARD_TUNAI_DETAIL_PROMPT comments for where this was verified directly
against assets/template.pdf).

No such parser existed anywhere in the codebase before this -- the
parenthetical text was previously discarded entirely (postprocessing.
normalize_currency: `text.split("(")[0]`) rather than read.

Grammar: satuan (0-9) x belasan (11-19) x puluhan (tens) x ratus (hundreds)
x ribu (thousands) x juta (millions) x miliar (billions) -- the standard,
regular Indonesian number-word grammar. Deliberately does NOT try to parse
free-form prose beyond a number phrase (a trailing "rupiah" is stripped, but
anything else unrecognized makes the whole parse fail -- returning None,
never a partial/guessed number).
"""
import re

_SATUAN = {
    "nol": 0, "satu": 1, "se": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
}
_BELASAN = {
    "sepuluh": 10, "sebelas": 11, "duabelas": 12, "tigabelas": 13,
    "empatbelas": 14, "limabelas": 15, "enambelas": 16, "tujuhbelas": 17,
    "delapanbelas": 18, "sembilanbelas": 19,
}
_PULUHAN_MULTIPLIER = 10   # "dua puluh" = 2 * 10
_SCALES = [
    ("miliar", 1_000_000_000),
    ("milyar", 1_000_000_000),
    ("juta", 1_000_000),
    ("ribu", 1_000),
    ("ratus", 100),
    ("puluh", 10),
]

_IGNORE_WORDS = {"rupiah", "rp", "rp.", "-", ","}

# Common fused "se-" + scale contractions ("seribu rupiah" = Rp1.000, not
# "satu ribu" spelled out) -- expanded to their two-word equivalent BEFORE
# the main grammar runs, so _SCALES/_parse_group only ever have to handle
# one spelling of each scale word.
_SE_SCALE_EXPANSION = {
    "seribu": ["satu", "ribu"],
    "sejuta": ["satu", "juta"],
    "semiliar": ["satu", "miliar"],
    "semilyar": ["satu", "miliar"],
}


def _normalize_words(text):
    text = text.lower().strip()
    text = re.sub(r"[.,]", " ", text)
    raw_words = [w for w in re.split(r"\s+", text) if w and w not in _IGNORE_WORDS]
    words = []
    for w in raw_words:
        words.extend(_SE_SCALE_EXPANSION.get(w, [w]))
    return words


def _parse_group(words):
    """Parse a sequence of words representing a number < 1000 (no ribu/juta/
    miliar scale words -- those are handled by the caller). Returns
    (value, words_consumed) or (None, 0) if the words don't form a valid
    small-number phrase at all."""
    if not words:
        return None, 0

    # "belasan" (11-19) is a single fused word in standard spelling
    # ("sebelas", "duabelas", ...) -- check first since it's unambiguous.
    if words[0] in _BELASAN:
        return _BELASAN[words[0]], 1

    total = 0
    i = 0
    n = len(words)

    # ratus (hundreds): "X ratus" or bare "seratus"
    if words[i] == "seratus":
        total += 100
        i += 1
    elif i < n and words[i] in _SATUAN and i + 1 < n and words[i + 1] == "ratus":
        total += _SATUAN[words[i]] * 100
        i += 2

    # puluh (tens) / belas (teens): "X puluh" or bare "sepuluh" (already
    # handled above via _BELASAN only if it's the FIRST word; "seratus
    # sepuluh" needs it here), a fused belasan word ("duabelas"), OR the
    # more common modern two-word spelling "X belas" ("lima belas" = 15,
    # "enam belas" = 16, ...).
    if i < n and words[i] == "sepuluh":
        total += 10
        i += 1
    elif i < n and words[i] in _BELASAN:
        total += _BELASAN[words[i]]
        i += 1
    elif (i < n and words[i] in _SATUAN and words[i] != "se"
          and i + 1 < n and words[i + 1] == "belas"):
        total += _SATUAN[words[i]] + 10
        i += 2
    elif i < n and words[i] in _SATUAN and i + 1 < n and words[i + 1] == "puluh":
        total += _SATUAN[words[i]] * _PULUHAN_MULTIPLIER
        i += 2

    # satuan (units): remaining single digit word
    if i < n and words[i] in _SATUAN and words[i] != "se":
        total += _SATUAN[words[i]]
        i += 1

    if i == 0:
        return None, 0
    return total, i


def parse_terbilang(text):
    """Parse an Indonesian spelled-out number string into an integer.
    Returns None if the text is empty or doesn't parse as a clean number
    phrase (never a partial/best-guess value -- an inconsistency check
    needs a real number or an honest "couldn't read it", not a guess).

    Examples: "lima juta rupiah" -> 5000000, "sepuluh ribu" -> 10000,
    "dua ratus lima puluh ribu rupiah" -> 250000, "seratus" -> 100."""
    if not text or not str(text).strip():
        return None
    words = _normalize_words(str(text))
    if not words:
        return None

    total = 0
    i = 0
    n = len(words)
    while i < n:
        # find the next scale word (ribu/juta/miliar/ratus/puluh) and parse
        # the group of words before it as that scale's multiplier
        scale_idx, scale_value = None, None
        for j in range(i, n):
            for name, value in _SCALES[:4]:  # miliar/milyar/juta/ribu only here
                if words[j] == name:
                    scale_idx, scale_value = j, value
                    break
            if scale_idx is not None:
                break

        if scale_idx is None:
            # no more ribu/juta/miliar ahead -- parse the rest as a bare
            # <1000 group (may itself contain ratus/puluh) and stop
            group_val, consumed = _parse_group(words[i:])
            if group_val is None or consumed != (n - i):
                return None  # leftover, unparseable words -- fail honestly
            total += group_val
            i = n
            break

        group_words = words[i:scale_idx]
        if not group_words:
            group_val = 1  # bare "ribu"/"juta" implies "satu ribu"/"satu juta"
            consumed = 0
        else:
            group_val, consumed = _parse_group(group_words)
            if group_val is None or consumed != len(group_words):
                return None
        total += group_val * scale_value
        i = scale_idx + 1

    # Every loop iteration above either returns None on any unparseable
    # leftover or fully consumes its slice (i reaches n) -- reaching here
    # means the WHOLE word list validly parsed, so `total` (including a
    # legitimate 0 for "nol") is trustworthy, not a default/unset value.
    return total
