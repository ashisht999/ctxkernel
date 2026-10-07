# 2026-10-01: name reach, ranked context, automatic failure notes

Same data and metric as [the first eval](2026-09-30-first-eval.md): 8 real
Claude Code sessions, about 550 scored steps, evidence recall. All runs here are
**offline** (no model). The Jev runs were blocked by the permission system,
because they send session content to an external API; see "Open" below.

## 1. Name reach fixed the candidate problem

Every node is now indexed by the names in its full content (blobs included).
Candidates = typed links from recent work **plus** any old node that shares a
rare name (IDF-weighted) with the agent's latest calls, its plan, the user's
input or the open task.

| | before | after |
|---|---|---|
| lossy steps where the needed item was among the candidates | 37% | **82%** |
| recall with a perfect scorer over the candidates | 0.930 | **0.961** |
| recall with a perfect scorer over any past item (ceiling) | 0.974 | 0.974 |

The scorer is now what matters: 0.914 (today's default) → 0.961 is available
to a scorer that ranks well.

## 2. Ranked context vs recency, on the same budget

`recency` spends the whole target on the recent tail (the fair baseline;
today's default `null` leaves 25% of the target unused). Ranked policies keep
a share for the recent tail and fill the rest in score order (threshold 0).

| policy | recall | steps with no loss | context |
|---|---|---|---|
| null (today) | 0.914 | 76.6% | 22.0k |
| recency, same budget | 0.932 | 80.2% | 27.5k |
| ranked by prior, 35% tail | 0.929 | 78.9% | 26.1k |
| **ranked by prior, 60% tail** | **0.942** | **81.5%** | **26.6k** |

The model-free prior beats recency by +0.010 on fewer tokens. 50 vs 150
candidates made no difference, so Jev requests can stay small. The gap to 0.961
is the headroom for a better scorer.

## 3. Automatic failure notes

After each tool result the background run asks "did this step fail?" and a
confident yes becomes a pinned failure note, quoted from the step's own output.
The baseline judge is a failure-pattern regex; Jev plugs into the same slot.

- 16 notes from 180 calls on this project's sessions; in a sample of 12, about
  10 were real failures (exit 1, tracebacks, failing tests, rejected commands).
- Evidence recall: +0.001 for +660 tokens. That is expected, because the notes
  exist to stop a failed approach being retried, and this metric does not see
  that. It needs the tier-2 / agent-run eval.
- Bug found and fixed: a call could be judged before its result arrived, then
  never again.

## Open

- **Jev ranking and judging on real sessions**: blocked from sending session
  content externally. Options: the user runs it, allows it, or it runs on public
  trajectories instead.
- **Tier 2** (does the agent act the same?) needs a model call per step; the same
  data question applies. Public SWE-agent / OpenHands trajectories avoid it and
  come with task outcomes.
- **Task-boundary inference** is not built. It needs a model judgment to be
  worth anything, and could not be evaluated today.
