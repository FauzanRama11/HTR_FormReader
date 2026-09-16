"""
config.py
=========
V19 -- centralized app configuration. Currently only Gemini token pricing +
the USD/IDR conversion rate (task spec section 8: "pricing configuration
must be easy to update when Gemini pricing changes" -- NOT hardcoded
throughout the app). extractors.py reads these constants, never hardcodes a
number itself -- update the numbers here when Gemini's published pricing or
the exchange rate changes, nothing else needs to change.

NOTE: "gemini-3.8-flash" is the only Gemini model id in extractors.
ENGINE_LABELS (gemini-2.5-flash/mistral-ocr-4-1 removed per explicit user
request) -- it's a forward-looking/placeholder model id with no published
pricing yet, so it falls through to GEMINI_DEFAULT_PRICING below until
Google publishes real numbers for it. Values below are best-effort at time
of writing and NOT re-verified against Gemini's live pricing page in this
sandbox -- confirm against https://ai.google.dev/gemini-api/docs/pricing
before trusting cost figures in production.
"""

# USD per 1,000,000 tokens. "output_per_1m" covers BOTH candidate output
# tokens AND thinking tokens (task spec section 8's cost formula sums them
# before applying this rate). Empty until Gemini publishes real
# "gemini-3.8-flash" pricing -- GEMINI_DEFAULT_PRICING below is used instead
# until then.
GEMINI_PRICING = {}

GEMINI_DEFAULT_PRICING = {"input_per_1m": 0.30, "output_per_1m": 2.50}

USD_IDR_RATE = 16300.0

# Hosted Qwen3-VL test ("qwen3_vl" engine, extractors.run_qwen3_vl_hosted).
# Env vars QWEN_MODEL/HF_REQUEST_TIMEOUT_S override these at call time
# (extractors.py never hardcodes either) -- these are just the defaults when
# those env vars are unset.
# V19f: aligned to the same 4B model validated for the local engine
# (vlm.DEFAULT_LOCAL_CANDIDATES / models/Qwen3-VL-4B-Instruct) instead of the
# smaller 2B model, per explicit user policy that local and hosted Qwen3-VL
# must be kept in sync going forward. Confirm this exact id is actually
# resolvable through the configured HF Inference Provider before relying on
# it in production -- it was not live-verified against the hosted API in the
# session that made this change.
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-4B-Instruct"
HF_REQUEST_TIMEOUT_S = 60
