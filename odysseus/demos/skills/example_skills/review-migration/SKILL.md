---
description: How to review a database migration before it is merged
---
# Reviewing a database migration

Read the migration file, then check these four points, in this order.
Report each on its own line as PASS or FAIL, with one short reason:

1. Reversible: is there a matching down migration (the same name with
   .down.sql) that undoes it?
2. Locking: does it add a NOT NULL column without a DEFAULT to an
   existing table? That fails on tables with rows, or locks them.
3. Indexes: does every new foreign key column have an index?
4. Data: does it delete or rewrite existing rows?

Format each line as:

    <n>. <name>: PASS|FAIL - <reason>

End with exactly one of these lines:

    Verdict: APPROVE
    Verdict: REQUEST CHANGES

Request changes if any check fails. Do not edit the migration yourself;
the author fixes it.
