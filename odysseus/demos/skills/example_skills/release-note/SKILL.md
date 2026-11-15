---
description: How to write a release note for a change in this project
---
# Writing a release note

Add the new entry to RELEASE_NOTES.md, directly under the
"# Release notes" heading, in exactly this format:

    ## <version> - <YYYY-MM-DD>
    ### Fixed
    - <what was wrong, in words a user understands> (<file changed>)
    Upgrade: <what a user must do, or "no action needed">

Rules:
- Use the version and date given in the task. Never invent them.
- One line per change, past tense, no code or jargon.
- Put the changed file name in brackets at the end of the line.
- Always include the Upgrade line, even when nothing is needed.
- Never edit or reorder older entries.
