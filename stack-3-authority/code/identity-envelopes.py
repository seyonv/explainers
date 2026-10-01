"""Authority only narrows: a tiny identity-envelope check.

Scopes are illustrative, built for the series' "Tuesday run":
a coding agent fixing test_parse_due_date in invoice-tools.
Python 3.10, standard library only.
"""

USER = frozenset({
    "repo:read", "repo:write", "tests:run", "git:push-branch",
    "git:push-main", "email:send", "secrets:read",
})


def narrow(parent, wanted):
    """A stage may keep or drop scopes. It may never add one."""
    extra = set(wanted) - parent
    if extra:
        raise PermissionError(f"broadens: {sorted(extra)}")
    return frozenset(wanted)


def allowed(envelope, action):
    """Enforcement point: outside the model, checked per action."""
    return action in envelope


if __name__ == "__main__":
    task = narrow(USER, {"repo:read", "repo:write", "tests:run",
                         "git:push-branch"})
    step6 = narrow(task, {"repo:read", "tests:run"})
    for name, env in [("user", USER), ("task", task),
                      ("step 6 pytest", step6)]:
        print(f"{name:14} {len(env)} scopes")

    # Untrusted text in a tool result asks for more authority.
    print("push to main?", allowed(task, "git:push-main"))
    try:
        narrow(step6, {"repo:read", "tests:run", "secrets:read"})
    except PermissionError as e:
        print("widen step 6:", e)

    # A broad service identity acting for the same user.
    bot = frozenset({"repo:read", "repo:write", "tests:run",
                     "git:push-main", "secrets:read", "org:admin"})
    print("bot alone   ", len(bot), "scopes, leaks",
          sorted(bot - task))
    print("bot & task  ", len(bot & task), "scopes, leaks",
          sorted((bot & task) - task))
