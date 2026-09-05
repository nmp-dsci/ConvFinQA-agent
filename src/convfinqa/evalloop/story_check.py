"""CI check: the published page still describes the record it claims to.

`docs/optimization/index.html` is committed, so it can go stale — someone
promotes a version, forgets to rebuild, and the public write-up quietly
describes a champion that is no longer the champion. A page that can drift from
the registry is a claim, and the whole point of generating it was that it should
not be one.

So this compares the committed `story.json` against the registry and the
tracking store, and fails when they disagree. It makes no API calls and reads
only committed files plus whatever tracking store is present, which is what
lets it run on every pull request.
"""

from __future__ import annotations

import json
import sys

from convfinqa.evalloop.story import DOCS_DIR, SDK_PAGE, STORY_PATH


def problems() -> list[str]:
    """Everything stale about the committed page. Empty means it is current."""
    from convfinqa.tracking import registry

    out: list[str] = []
    if not STORY_PATH.exists():
        return []  # nothing published yet is not staleness
    story = json.loads(STORY_PATH.read_text())

    champion = registry.champion()
    if story.get("champion") != champion:
        out.append(
            f"story.json names champion {story.get('champion')!r} but the registry "
            f"says {champion!r} — rebuild with `convfinqa-evalloop story`"
        )

    page = DOCS_DIR / "index.html"
    if not page.exists():
        out.append(f"{page} is missing — the published page was never built")
    else:
        text = page.read_text()
        if champion and f'"champion": "{champion}"' not in text:
            out.append(
                f"{page.name} does not carry the current champion {champion!r} in "
                "its embedded record — it was built from an older story.json"
            )

    # The Agent SDK page is built from the same story.json by the same command,
    # so it goes stale in exactly the same ways — a missing page, or one whose
    # embedded record names a different sdk_champion from the registry.
    sdk_champion = registry.sdk_champion()
    if story.get("sdk_champion", None) != sdk_champion:
        out.append(
            f"story.json names sdk_champion {story.get('sdk_champion')!r} but the "
            f"registry says {sdk_champion!r} — rebuild with `convfinqa-evalloop story`"
        )
    sdk_page = DOCS_DIR / SDK_PAGE
    if not sdk_page.exists():
        out.append(f"{sdk_page} is missing — the Agent SDK page was never built")
    else:
        sdk_text = sdk_page.read_text()
        if sdk_champion and f'"sdk_champion": "{sdk_champion}"' not in sdk_text:
            out.append(
                f"{sdk_page.name} does not carry the current sdk_champion "
                f"{sdk_champion!r} in its embedded record — it was built from an "
                "older story.json"
            )
        n_sdk = sum(
            len(c.get("experiments", [])) for c in story.get("sdk_campaigns", [])
        )
        if f'"n_sdk_experiments": {n_sdk}' not in sdk_text:
            out.append(
                f"{sdk_page.name} records a different number of SDK experiments "
                f"from story.json ({n_sdk}) — rebuild with `convfinqa-evalloop story`"
            )

    published = DOCS_DIR / "story.json"
    if published.exists() and published.read_text() != STORY_PATH.read_text():
        out.append(
            "docs/optimization/story.json differs from evaluation/story.json — "
            "the page and the record were built from different data"
        )

    n_experiments = sum(
        len(c.get("experiments", [])) for c in story.get("campaigns", [])
    )
    # Only cycle-driven promotions imply an experiment record. Promotions made
    # before the loop existed, and deliberate rollbacks, legitimately have none —
    # flagging those would make the check cry wolf on the very history it exists
    # to protect.
    from_cycle = [
        ev
        for ev in story.get("lineage", [])
        if str(ev.get("actor", "")).startswith("evalloop-cycle")
    ]
    if from_cycle and not n_experiments:
        out.append(
            f"story.json records {len(from_cycle)} promotion(s) made by the cycle "
            "but no experiments — the campaign query returned nothing, which "
            "usually means the page was built against a different tracking store "
            "from the one holding the runs"
        )
    return out


def main() -> int:
    """Exit non-zero when the published page is stale."""
    found = problems()
    if not found:
        print("story: published page is current")  # noqa: T201
        return 0
    print("story: the published page is stale\n")  # noqa: T201
    for line in found:
        print(f"  - {line}")  # noqa: T201
    print("\nRebuild it with: uv run convfinqa-evalloop story")  # noqa: T201
    return 1


if __name__ == "__main__":
    sys.exit(main())
