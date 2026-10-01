"""The Tuesday run as an RL training trajectory: which tokens get loss,
what advantage they get, and why compaction and retokenization matter.

Everything about the Tuesday run (token counts, the group of rollouts,
their rewards and lengths) is illustrative, written for this card.
The mechanisms follow Wolfe, "Agentic RL: Frameworks and Best
Practices" (2026): the action mask (Agent-R1), GRPO group advantages,
task-level advantage normalization (AgentRL, using the post's own toy
inputs), retokenization drift (Agent-R1) and compaction splitting a
rollout into sub-traces (GLM-5.2, as quoted in the post).
"""
import math

import numpy as np

# (segment, tokens, written by the agent?)  -- illustrative counts
TUESDAY = [
    ("system prompt + task", 1200, False),
    ("steps 1-5: thinking + tool calls", 750, True),
    ("steps 1-5: tool outputs", 2400, False),
    ("step 6: call pytest", 40, True),
    ("step 6: pytest output, exit 1", 560, False),
    ("step 7: compaction summary", 60, False),
    ("step 8: thinking + tool call", 120, True),
    ("step 8: tool output", 300, False),
    ("step 9: 'Done ... all tests pass'", 30, True),
]


def action_mask_share(segments):
    """Tokens that get loss (mask 1) out of all tokens in the trace."""
    total = sum(n for _, n, _ in segments)
    agent = sum(n for _, n, mine in segments if mine)
    return agent, total


def grpo_advantages(rewards, eps=1e-8):
    """(r - group mean) / group std, one number per rollout."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def task_normalize(rewards, prompt_ids, task_ids, mask):
    """AgentRL: GRPO advantage per trajectory, broadcast to agent
    tokens with the action mask, then z-scored within each task."""
    rewards, mask = np.asarray(rewards, float), np.asarray(mask, float)
    prompt_ids, task_ids = np.asarray(prompt_ids), np.asarray(task_ids)
    adv = np.zeros_like(rewards)
    for p in np.unique(prompt_ids):
        g = prompt_ids == p
        adv[g] = grpo_advantages(rewards[g])
    tok = adv[:, None] * mask
    out = np.zeros_like(tok)
    for t in np.unique(task_ids):
        rows = task_ids == t
        vals = tok[rows][mask[rows].astype(bool)]
        z = (tok[rows] - vals.mean()) / (vals.std() + 1e-8)
        out[rows] = z * mask[rows]
    return adv, out


def greedy_tokenize(text, vocab):
    """Longest-match tokenizer: a stand-in for re-encoding text."""
    out, i = [], 0
    while i < len(text):
        for j in range(len(text), i, -1):
            if text[i:j] in vocab:
                out.append(text[i:j])
                i = j
                break
        else:
            out.append(text[i])
            i += 1
    return out


def sub_traces(length, window):
    """How many trainable pieces compaction cuts a rollout into, if
    it compacts each time the window fills (a simplification)."""
    return math.ceil(length / window)


if __name__ == "__main__":
    agent, total = action_mask_share(TUESDAY)
    print(f"Tuesday trace: {total:,} tokens, {agent:,} written by "
          f"the agent ({agent / total:.1%}) get loss")

    # four rollouts of the same task; the Tuesday run is rollout 1
    rewards = [0, 1, 1, 0]
    adv = grpo_advantages(rewards)
    print("group rewards", rewards, "-> advantages", adv.round(2))
    print(f"every one of the {agent} agent tokens in the Tuesday run "
          f"gets advantage {adv[0]:+.2f}")

    # Wolfe's toy inputs for task-level normalization (AgentRL)
    r = [1.0, 0.0, 0.5, 1.0, 0.0, 1.0]
    p = [0, 0, 1, 1, 2, 2]
    t = [0, 0, 0, 0, 1, 1]
    m = [[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1],
         [0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0]]
    traj, norm = task_normalize(r, p, t, m)
    print("trajectory advantages", traj.round(2))
    print("task-normalized, first agent token of each row",
          norm[:, 1].round(2))

    # retokenization drift: the policy sampled 'pass' + 'ed'
    vocab = {"pass", "ed", "passed", " all", " tests", " "}
    sampled = [" all", " tests", " ", "pass", "ed"]
    text = "".join(sampled)
    again = greedy_tokenize(text, vocab)
    print("sampled  ", sampled)
    print("re-encoded", again, "same?", sampled == again)

    # compaction: same prompt, different numbers of trainable traces
    lengths = [24_000, 70_000, 45_000, 110_000]  # tokens per rollout
    window = 32_000
    print("sub-traces per rollout",
          [sub_traces(n, window) for n in lengths])
