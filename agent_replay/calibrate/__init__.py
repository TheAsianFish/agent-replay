"""Judge calibration: golden sets, human labels and agreement with the judge.

A judge check gates CI only when a calibration report shows that it agrees
with human labels (PLAN.md decision 5). This package builds the golden set
(:mod:`.golden`) and computes the report.
"""
