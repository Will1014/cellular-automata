---
type: decision-rule
project: PRJ-013 Cellular Automata
date: 2026-10-12
status: approved by Will 2026-10-12, fixed before any run
---

# Gate 0: rule for keeping or dropping each metric

Written before any of the three new measures has been computed on a single
rule. Plan section 4 sets the gate; this fixes how it is scored.

## What is being tested

Three measures, each one number per rule:

- **Free will (entropy rate, h).** How much is still left to guess about
  the next cell once you know the whole past. Fair coin: 1 bit. Anything
  that repeats: 0.
- **Fate (excess entropy, E).** How much the past tells you about the
  future. Fair coin: 0. All zeros: 0. Alternating 0101...: 1 bit. Plan's
  claim: noise is high free will and low fate, structure is some free will
  and high fate.
- **Transition coefficient (C)**, after Zenil (2010, Complex Systems 19).
  Start from 40 neighbouring initial conditions (each differs from the next
  by one cell, Gray-code order), run each, compress each run, and take the
  mean jump in compressed length between neighbours, divided by run length.
  Repeat for run lengths 50, 100, ..., 600. C is the slope of that sequence.

**Prediction, from the literature review:** class IV has less free will,
more fate and a higher C than class III.

**Wolfram's classes are used only as an answer key.** They are cases where
we already know which behaviour is structure and which is noise. Worlds in
the framework are never sorted into these classes; they are placed on
continuous axes, and the empty regions of that map are a search target
(`Vault/Reference/Cellular automata - search the empty regions of the map.md`).

Worth saying before the run: Zenil's own Table 2 already puts class III rule
22 (C = 2.5) far above class IV rules 54, 110 and 41 (0.21 to 0.27), so C
starts with one known exception against it.

## Fixed settings

- **Rules:** the 88 equivalence classes, one representative each, with
  Martinez (2013) labels as in `wolfram_classes.py`. Not all 256: mirror and
  complement copies behave identically, so counting them would treat one
  behaviour as two to four independent data points. A side check confirms
  that equivalent rules score the same.
- **Runs for free will and fate:** width 10,000, periodic, 2,000 steps, the
  first 1,000 discarded, random starts from seeds 0 to 4 (numpy PCG64). That
  gives 10 million windows per seed. Estimated per seed, then the per-rule
  mean over seeds.
- **Estimator:** block entropy H(L), in bits, of length-L windows along each
  kept row, pooled over rows. h(L) = H(L) - H(L-1), E(L) = H(L) - L h(L).
- **Look-back for the decision: L = 16, the same for every rule, fixed
  here.** It is the longest the data supports for the noisiest rules (a
  coin-flip world sees each of its 65,536 patterns about 150 times) and
  it is longer than rule 110's 14-cell background. Choosing a look-back per
  rule, or after seeing results, is ruled out.
- **Fate curve, recorded, not decisive:** H(L) for L = 1 to 24, kept only
  while the number of distinct patterns seen is at most 1% of the samples.
  A curve still rising at its last trusted L is flagged as long memory.
- **C:** Zenil's settings, n = 40, t = 50 to 600 in steps of 50. Start j
  (j = 1 to 40) is the Gray code of j as a 6-cell block centred on a zero
  background, wide enough that nothing wraps (6 + 2t + 2 cells). Each
  t-step space-time diagram is bit-packed and compressed with zlib level 9.
  c(t) = mean absolute jump in compressed length between starts j and j+1,
  divided by t; C = least-squares slope of c(t) against t's index 1 to 12.
  Zenil used Mathematica's compressor, so only ranks are comparable with
  his Table 2. Deterministic, so no seed.
- **Comparison measures,** recomputed on the same runs: zlib compression
  ratio of the kept rows for free will and fate, and damage spreading (flip the centre cell, fraction of cells differing at the last step) for C.

## Three contrasts, nine tests

Contrasts as in the existing sweep: IV vs III (4 against 11), IV vs rest
(4 against 84), III+IV vs I+II (15 against 73). The score is AUC, the
chance that a randomly chosen positive rule outranks a randomly chosen
negative one, oriented so the predicted direction is above 0.5.

- **p-values:** from the shuffled-label null. One-sided for IV vs III,
  where the review predicts a direction. Two-sided for the other two, where
  it doesn't (against class I and II, class IV should have more free will,
  not less). Clarified 2026-10-12 before any result existed. IV vs III is
  exact (all 1,365 ways of choosing 4 rules from 15). The other two use
  100,000 shuffles, seed 0.
- **Multiple tests:** Holm correction across all nine tests at 0.05.
  Strict on purpose: at best a measure faces a threshold of 0.05 / 9 =
  0.0056. For IV vs III that means at most 7 of the 1,365 labellings can
  do as well, so separation has to be close to perfect.

## The rule

A measure is **kept** only if all three hold:

1. **It separates IV from III.** The IV vs III test survives Holm. This is
   the plan's prediction and the only test that decides.
2. **It survives dropping 41 and 106.** On 54 and 110 against class III,
   the oriented AUC is at least 0.75. With 2 positives no test can reach
   significance (the smallest possible p is 1/78), so this is a direction
   and size check only.
3. **It does better than the measure it would replace.** Its oriented IV vs
   III AUC is higher than zlib's (for free will and fate) or damage
   spreading's (for C) on the same runs. The old measure gets whichever
   direction suits it better, so the comparison is never rigged in the new
   measure's favour.

Outcomes:

- **Keep:** passes 1 to 3.
- **Keep as a replacement:** passes 1 and 2, fails 3, and Spearman |rho|
  with the old measure across the 88 rules is 0.9 or more. It is the same
  information in a form that carries to stochastic lattices, which zlib
  and damage spreading do not.
- **Drop:** anything else.

IV vs rest and III+IV vs I+II are reported, with corrected p-values, but
they do not decide.

## Checks run before the result is read

- **Known answers for the estimators:** fair coin gives free will about 1
  and fate about 0, a period-2 pattern gives 0 and 1, period 3 gives 0 and
  log2 3, period 14 gives 0 and log2 14.
- **Null against null:** 1,000 fake measures of pure noise go through the
  same rule. Rule 1 must pass at most about 5% of them. The IV vs III null
  must centre on exactly 0.5. If either is off, the test gets fixed before
  any result is read.
- **Symmetry:** all 256 rules are scored once, and each rule matches its
  representative within seed noise.

## Recorded with every result

Seed list, generator, settings, code commit (with a flag if the tree was
dirty), package versions and both clocks. One output file per rule per
measure, written as each finishes. The aggregation and null test run as a
separate step, reading those files.

## Limits, stated now

Four class IV rules, two of them contested. Fate at L = 16 is a lower bound:
any memory longer than 16 cells is invisible to the decision, and only the
curve hints at it. Passing is necessary, not
sufficient: it says little about stochastic lattices. Failing on
elementary rules is a strong reason to drop a measure, because these are
the cases it is supposed to handle best.
