# 01 - Auth Matrix

huggingface_hub 1.11.0

## Test 1: Can each token upload a file?

| Token | Role | Upload result |
|---|---|---|
| read | read | FAIL 403 |
| write | write | OK |

A read token is blocked (403). A write token is allowed (OK).

## Test 2: Which token wins when several are present?

Setup: the token saved on my PC is a write token, and the
HF_TOKEN environment variable holds a read token.

| What I did | Tokens available | Role that was used |
|---|---|---|
| Passed token=... in code | the token I passed | write |
| Passed nothing, HF_TOKEN is set | HF_TOKEN + saved token | read |
| Passed nothing, HF_TOKEN is not set | saved token only | write |

Rule: token=... wins, then HF_TOKEN, then the saved token.
