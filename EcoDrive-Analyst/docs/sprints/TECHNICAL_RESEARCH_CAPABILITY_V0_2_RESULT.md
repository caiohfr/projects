# Technical Research Capability v0.2 — Live Evidence Result

Date: 2026-09-21

## Implementation

The accepted v0.1 graph was extended only with deterministic source
classification and application-matching nodes. Live discovery uses DDGS;
technical claim extraction uses GPT-5.6 Terra through the Responses API.
Source authority, application match, identity confidence, and conflict
resolution remain deterministic Python policy decisions.

Focused deterministic tests: 26 passed, including 7 classification subtests.
Main application regression: 1847 passed, 174 subtests passed.

## BMW benchmark

- configurations: 15
- searches: 45
- discovered results: 225
- audited unique request/source rows: 213
- primary technical: 36
- strong technical: 4
- discovery-only: 173
- application matches: 5 partial, 13 mismatch
- result status: 4 partially supported, 11 insufficient evidence
- direct/strong/weak identities: 0
- unresolved identities: 15
- candidate groups unresolved: 9

The live benchmark is operational, but the retrieved sources did not contain
sufficient exact-application evidence to prove a transmission designation for
any of the 15 configurations. The policy therefore left every identity
unresolved. This is the intended conservative outcome and does not justify
rerunning the transmission grouping signal experiment yet.

## Safety

No EcoDrive database was modified. SHA256 before and after:

- production: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- QA: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- staging candidate: `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`

The full audit and required status block are in
`artifacts/technical_research/BENCHMARK_SUMMARY_V02.md`.

