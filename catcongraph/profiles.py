from __future__ import annotations

from typing import Any, Mapping


STANDARD = "standard"
# Backward-compatible symbol used by established scientific modules. Its value
# is now the public, version-free name; historical YAML spellings remain input
# aliases below so an existing project is not forced onto a different branch.
MANUSCRIPT_LEGACY = STANDARD
GENERAL = "general"


# Exact project identifiers with contact-unit definitions validated in the
# reported reference analyses.  Aliases are retained for the corresponding
# experimental/template identifiers already supported by the workflow.  Exact
# matching is deliberate: a new project such as ``AqTMK2`` must be treated as
# a new enzyme rather than silently receiving the AqTMK rules.
VALIDATED_CONTACT_UNIT_TARGETS = {
    "gac": "GAC",
    "aqtmk": "AqTMK",
    "2pbr": "AqTMK",
    "ptp1b": "PTP1B",
    "6b90": "PTP1B",
}


def validated_contact_unit_target(target_id: object) -> str | None:
    """Return the reported reference case matched by an exact target ID."""

    return VALIDATED_CONTACT_UNIT_TARGETS.get(str(target_id or "").strip().casefold())


def workflow_profile(config: Mapping[str, Any]) -> str:
    """Return the project's explicitly selected calculation profile.

    ``standard`` preserves the established ligand contact-unit grouping and topology
    contract for existing projects. ``general`` remains an explicit opt-in for
    exploratory analyses. The GUI is only a front end and must not silently
    replace a project's scientific calculation profile.
    """
    candidates = [
        (config.get("workflow", {}) or {}).get("profile"),
        (config.get("project", {}) or {}).get("profile"),
        (config.get("compatibility", {}) or {}).get("profile"),
    ]
    raw = next(
        (str(value).strip().lower() for value in candidates if value not in {None, ""}),
        MANUSCRIPT_LEGACY,
    )
    aliases = {
        "manuscript_legacy": MANUSCRIPT_LEGACY,
        "catcongraph_studio_v1": MANUSCRIPT_LEGACY,
        "standard": MANUSCRIPT_LEGACY,
        "legacy": MANUSCRIPT_LEGACY,
        "manuscript": MANUSCRIPT_LEGACY,
        "article": MANUSCRIPT_LEGACY,
        "article_reproduction": MANUSCRIPT_LEGACY,
        "manuscript_strict": MANUSCRIPT_LEGACY,
        "general_v1": GENERAL,
        "generalized": GENERAL,
        "batch": GENERAL,
    }
    profile = aliases.get(raw, raw)
    if profile not in {MANUSCRIPT_LEGACY, GENERAL}:
        raise ValueError(
            "workflow.profile must be 'standard' or 'general'. "
            f"Got: {raw!r}"
        )
    return profile


def is_general_profile(config: Mapping[str, Any]) -> bool:
    return workflow_profile(config) == GENERAL
