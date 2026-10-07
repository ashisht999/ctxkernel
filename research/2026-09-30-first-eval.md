# 2026-09-30: first evidence-recall eval on real traces

**Data:** 8 of the author's own Claude Code sessions (4 projects), 533 scored action
steps. Local only, except for the Jev run on two sessions from this project,
where the API key was redacted before sending.

**Metric** (`python -m ctxkernel.eval`): at each step where the agent acted, the
names its action used (paths, symbols, numbers) that it learned from earlier
history rather than from the user, and the fraction still visible in the context
the policy would have sent. Model-free, deterministic, a proxy (see the
`ctxkernel/eval.py` docstring).

## 1. Where the misses came from (null policy, before any fix)

| cause | share |
|---|---|
| name sat in a handle's blob, not its extract (mostly long `Bash` output → `DefaultCodec`) | 59% |
| aged out of the tail | 40% |
| removed by the cascade (superseded / invalidated / duplicate) | **0.3%** |

The cascade's "provable" removals almost never removed anything the agent later
used. That is the strongest evidence so far for the core idea.

## 2. Fix: names index on every extract

`codecs.names_index`: every extract gets the paths, file names and identifiers
it cut, interleaved by kind, capped at about 150 tokens, with a "+N more" count.
Codecs that cut on purpose opt out (`TestOutputCodec.index_names = False`).

On a fixed session (307cb966): recall 0.696 → **0.866**; steps with no loss
40.5% → **75.7%**; context +17% (9.3k → 10.8k).

Caveat: the metric counts names and the fix surfaces names, so part of the gain
is by construction. Whether the agent *acts better* needs the tier-2 eval (a
model predicts the next action from each context).

## 3. All 8 sessions, after the fix

| policy | recall | steps with no loss | context | raw | saved |
|---|---|---|---|---|---|
| null | 0.921 | 78.8% | 21.9k | 105.9k | 79.4% |
| proximity | 0.927 | 79.4% | 22.1k | 105.9k | 79.2% |

The real-trace trade-off is **about 79% saved at about 92% recall**, not the
synthetic 97.7%. Short sessions lose nothing; the largest (1,222 messages)
keeps 96.4% of the names at about 27k context vs 130k raw.

## 4. Jev, and why it didn't help (yet)

Two sessions, 125 Jev runs at about 500 ms each, no failures: recall 0.866 →
0.866 and 0.886 → 0.891. Upper bounds with a perfect scorer:

| | recall | steps with no loss |
|---|---|---|
| null | 0.920 | 78.2% |
| perfect scorer over the graph's candidates | 0.930 | 79.7% |
| perfect scorer over any past item | **0.975** | **89.1%** |

In 63% of lossy steps the needed item was **not among the candidates**:
candidates are the 2-hop neighborhood of recent calls over reads, writes and
derived_from, and what the agent needs next is often an old item those links
never reach. The scorer is not the bottleneck; **reach** is.

## Next

1. **Mentions links:** an edge wherever two events share a name (from the same
   parse as the names index). Candidates can then reach old items that the recent
   work mentions. Re-measure the candidate-limited upper bound; it has to
   approach 0.975 before a scorer can matter.
2. Then re-run Jev.
3. Tier-2 eval (next-action agreement with an LLM) to check that recall gains
   are decision gains.
