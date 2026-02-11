# Empirical analysis of fairness criteria on COMPAS data
## Practical Implementation for Bachelor's Thesis

This repository contains the interactive visualization tool developed for my bachelor's thesis on fairness in classification systems. 
Through an empirical analysis of the COMPAS recidivism dataset, the thesis looks at the theoretical underpinnings of statistical non-discrimination criteria and shows how incompatible they are in practice.

## Thesis Context

### *Title: Die Herausforderungen bei der Gewährleistung von Fairness im Kontext von Klassifizierung: Theoretische und praktische Aspekte*

This implementation corresponds to Chapter 6: Empirical Analysis and Visualizations of COMPAS Data. It serves as the practical counterpart to the theoretical framework developed in earlier chapters, particularly:

- Chapter 3 — Formalization of Independence, Separation, and Sufficiency
- Chapter 4 — Proofs of incompatibility between these criteria 
- Chapter 5 — The COMPAS dataset and the fairness paradox

## Goal

This thesis' theoretical section shows that the three statistical nondiscrimination fairness criteria (Independence, Separation and  Sufficiency) cannot all be met at the same time.

This impossibility theorem is visualized by this tool. It doesn't suggest a fair classifier or try to "fix" COMPAS. Rather, it enables the user to observe empirically that:

- Different ROC regions are occupied by various demographic groups.
- A group shifts away from another when it moves toward a fairness criterion.
- All three requirements cannot be met at the same time by a single decision threshold.

## Dataset

The analysis uses the COMPAS Two-Year Recidivism dataset collected and published by ProPublica. This dataset has been central to the fairness discourse since 2016.

- File: datasets/compas-scores-two-years.csv
- Source: [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis)

#### *To install all the required libraries --> pip install -r requirements.txt*
