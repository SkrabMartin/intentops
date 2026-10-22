# Calibration and human-review recommendations

The Phase-2 TF-IDF + multinomial logistic classifier is frozen. This experiment
never refits it and never opens official test.csv. BANKING77 is English banking,
single-intent support data, not evidence about real-company ticket performance.
The overarching validation partition was already evaluated in Phase 2. This
internal development/audit split is not a new untouched benchmark.

## Predeclared protocol

Verify the pinned Phase-1 manifest and validation CSV bytes. Stratify validation
indices with sklearn train_test_split, seed 43: test_size=.6 leaves 800 calibration
examples; split the remaining 1201 with test_size=.5, seed 43, yielding 600 selection
and 601 audit examples. Sort by original source_row_id. Every class has support in
all partitions, with no overlaps or base training rows. Local manifests retain IDs,
per-class supports and membership hashes without text. Original files remain unchanged.

Fit one scalar T on calibration decision scores by minimizing mean negative log
likelihood with logsumexp. Optimize log(T) in [log(.05),log(20)] with SciPy bounded
minimize_scalar, xatol=1e-10, maxiter=500; optimizer failures are explicit.
Compute softmax(scores/T); T=1 must match existing predict_proba at atol=rtol=1e-12.
Positive T preserves top-category ranking; stable declared class order breaks ties.
This applies the scalar method described by [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html),
without claiming their neural-network results transfer to this classifier.

Freeze T before threshold selection. Use thresholds 0,.01,...,1, chosen demonstration
error budget .10 and at least 30 accepted selection examples. Accept means maximum
of the full calibrated distribution >= threshold. Choose maximum accepted count
subject to one-sided 95% Wilson error upper bound <= .10; smallest threshold on
ties. If none qualifies, null threshold means review all. Zero accepted means
undefined selective error/bound (JSON null), never zero risk.

For n accepted, e errors, p=e/n, z=1.6448536269514722:

```text
upper = (p + z²/(2n) + z*sqrt(p*(1-p)/n + z²/(4n²))) / (1 + z²/n)
```

This approximate binomial Wilson bound uses the one-sided z quantile described by
[NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
It assumes binomial-style independent errors. It is not simultaneous over the
threshold grid. Selection creates optimism; small counts and distribution shifts
limit interpretation. The separate internal audit is the post-selection check;
it still cannot guarantee new-company or out-of-domain performance.

`develop` persists frozen evidence before `audit` can run, scoring only calibration
and selection rows. `audit` verifies frozen identity and acquires an exclusive
audit.claim before scoring audit examples. Repeat attempts fail; failed attempts
remain claimed. No parameter changes in response to audit results. Existing roots
refuse redevelopment. A deliberately new experiment requires a reviewed protocol
and storage root; do not remove claims or rerun until favorable results appear.

## Metrics and observed native results

Native model tfidf-lr-7a1a49c4228bb52f, Python 3.12.13, sklearn 1.9.1.
Measured 2026-10-22: T=0.34489862499484253, threshold=.66,
policy review-d1995987cf7ca46e. Generated artifacts stay private/local.

| Subset | n | Top-1 accuracy (unchanged) | Log loss raw → scaled | Brier raw → scaled | ECE raw → scaled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration fitting | 800 | .856250 | 1.284347 → .478602 | .470687 → .201162 | .471589 → .029595 |
| Threshold selection | 600 | .840000 | 1.357877 → .573736 | .499280 → .232460 | .472931 → .036211 |
| Frozen internal audit | 601 | .833611 | 1.412941 → .616569 | .507250 → .240886 | .472348 → .041203 |

Log loss is mean -log(actual-class probability), floored at 1e-300 for finite
reporting. Brier is mean sum over classes of squared probability minus one-hot
target, without division by class count. ECE uses support-weighted absolute
mean top-score minus accuracy in ten fixed bins [0,.1),...,[.9,1]. Empty bins
contribute zero; their means are null. ECE is bin-dependent and sample-sensitive.
Calibration-subset improvement is in-sample evidence.

| Frozen .66 policy | Accepted / n | Errors | Coverage | Review rate | Selective error | One-sided 95% upper error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selection | 494 / 600 | 37 | 82.33% | 17.67% | 7.49% | 9.68% |
| Internal audit | 483 / 601 | 38 | 80.37% | 19.63% | 7.87% | **10.13%** |

**The audit upper bound exceeds the 10% demonstration budget.** The frozen policy
remains unchanged. No safety or production acceptance claim follows. Calibration
sharpens scores, including incorrect predictions; high scores remain possible for
wrong/unknown intents. Policy ready means integrity/compatibility verified, not
that the audit proved a safety objective. The dashboard shows the audit warning.

Local JSON retains reliability bins, the 101-point selection curve, supports,
identities and audit results. development.png plots coverage versus error and
threshold versus review rate. Neither policies, manifests nor plots are published.
See README and [serving.md](serving.md) for operator commands and safe fallback.
Phase 5 (batch/CSV, UI/deployment and performance polish) stays separate. Official
held-out evaluation requires separately authorized future release work.
