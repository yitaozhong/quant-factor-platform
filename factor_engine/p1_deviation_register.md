# P1 Deviation Register

Every place this library deliberately differs from the Open Source Asset Pricing reference
scripts, with the reason. Governed by the four correctness principles (no look-ahead,
correct formulas, NaN discipline, independent verification). Generated 2026-07-24;
maintained by hand from here on.

## A. Deviations from principled re-evaluation of script quirks

### Accruals
Accruals: engine = [ (dACT-dCHE) - (dLCT-dDLC-dTXP_fillna0) - dp ] / avg(at, at_l12), with all 12-month differences taken as calendar lags on the wide grid (NaN when the month-t-12 observation is missing). Reference script instead uses POSITION-based groupby(permno).shift(12) over gappy m_aCompustat rows, so across listing/coverage gaps its 'annual' change pairs month t with data more than 12 calendar months old (e.g. permno 28388 1966-06 differenced against 1965-01). The reference is wrong: Sloan 1996 defines a fiscal year-over-year change; comparing against arbitrarily stale observations is an unintended artifact of Stata-style row shifting. Correct definition: value(t) - value(t-12 calendar months), NaN if either side missing. Residual divergence vs oracle: ~15.4k cells where the oracle emits stale-lag values the engine correctly leaves NaN.

### AssetGrowth
AssetGrowth: We deviate from AssetGrowth.py's lag construction. The script's groupby(permno).shift(12) is positional and, across gaps in m_aCompustat, silently computes 'annual' growth from an asset value more than 12 calendar months old (15,556 oracle cells, 400/400 sampled confirmed as gap-bridges). Correct definition: AssetGrowth_t = (at_t - at_{t-12cal}) / at_{t-12cal}, NaN when the calendar t-12 observation does not exist, and NaN (not inf) when at_{t-12} == 0 — the zero guard matches the script's own L66-70, which we adopt; only the calendar-vs-positional lag semantics deviate.

### BrandInvest
BrandInvest: RESOLVED as a deliberately matched oracle quirk (no longer a deviation). The reference (Stata sequential replace) writes the xad/0.6 seed but then runs the recursion BC_t = 0.5*BC_{t-1} + xad_t unconditionally from each gvkey's second row over ZERO-initialized values, so the seed survives only when the gvkey's very first row has non-missing xad; for every other firm the effective seed is plain xad (not 1.667*xad). The engine previously preserved the paper-mandated seed at the first non-missing-xad year (registered here as a deviation); it now replicates the script's overwritten-seed behavior exactly — zero-init, conditional seed at row 0 only, unconditional i-from-1 recursion, NaN before the first non-missing xad and at missing-xad years, PLAIN division by at (at==0 with BC!=0 -> inf, kept: downstream xad0/inf -> 0.0 rows survive like the oracle). The ~580 pre-existing inf cells (at==0 chains) are identical on both sides.

### Cash
Cash: engine adopts the reference's announcement-date timing (quarter available from month(rdq), carried +0/+1/+2 months, newest rdq wins per gvkey-month, broadcast to permnos via the SignalMasterTable gvkey link, atq>0 required) — the script is RIGHT on timing and the engine's former datadate+3-month stamp was replaced. Deviation: the reference's Stata-style dedup tags singleton gvkey-rdq groups dup=0 and then keeps only dup==1, silently discarding every quarter whose gvkey-rdq appears exactly once in the panel — an unintended translation artifact, not a filter with economic meaning. Correct definition: one row per gvkey-rdq (first row with non-missing atq), singletons retained. Two residual classes vs the oracle, both attributable to retained singleton (gvkey, rdq) announcements: (a) 2,285 cells present in engine but absent in oracle (retained singleton quarters), and (b) 84 value-shift cells (max ~0.335) where a retained singleton announcement (newest rdq) overrides an older announcement window the oracle keeps. 0 missing-in-native. Both are expected and correct.

### ChAssetTurnover
ChAssetTurnover: We adopt the script's deliberate ppent forward-fill (ChAssetTurnover.py L49) inside temp = (rect+invt+aco+ffill(ppent)+intan)-(ap+lco+lo). We deviate in one place: when average net operating assets (temp + temp_l12)/2 equal zero, the script stores AssetTurnover = inf and can emit inf ChAssetTurnover (12 cells, e.g. permno 11460 1991-04); we define the ratio as NaN when its denominator is zero — an infinite turnover is not a value, it is an unguarded division. All lags are calendar-based in both implementations.

### ChInv
ChInv: engine uses strict 12-calendar-month lags (DELTA(invt,12), LAG(at,12) on the monthly grid); reference ChInv.py:38-40 uses positional groupby.shift(12) on deduped m_aCompustat rows, so across coverage gaps its 'lag' uses data 13 to 48+ months old (e.g. permno 29103 @ 1986-06 uses 1982-06 balance-sheet data). The reference is wrong: Thomas & Zhang (2002) and OSAP's Stata original define a 12-calendar-month change; positional shift across gaps mislabels multi-year changes as annual. Correct definition: ChInv_t = (invt_t - invt_{t-12cal}) / ((at_t + at_{t-12cal})/2), NaN when no m_aCompustat observation exists exactly 12 calendar months prior. The 15,586 across-gap oracle cells are masked as oracle_artifact in reconciliation.

### ChNNCOA
ChNNCOA: engine uses a 12-calendar-month DELTA of NNCOA/at on the monthly grid; reference ChNNCOA.py:54 uses positional groupby.shift(12) on deduped m_aCompustat rows, so across coverage gaps the 'twelve-month change' spans 24+ months (e.g. permno 18331 @ 1953-06 differences against 1951-06). The reference is wrong: Soliman (2008) Table 7 DeltaNCO is an annual change and OSAP's Stata original used calendar l12. Correct definition: ChNNCOA_t = temp_t - temp_{t-12cal}, temp = ((at-act-ivao)-(lt-dlc-dltt))/at, NaN when the t-12-calendar-month observation is absent. The 15,252 across-gap oracle cells are masked as oracle_artifact.

### ChTax
ChTax: engine = (txtq - txtq_l12cal) / at_l12cal with annual total assets 'at' from m_aCompustat as the denominator (matching the reference and recovering ~200k early-sample cells the quarterly-atq denominator would kill), lags calendar-based. Deviation: the reference leaves the division unguarded and its save step keeps +/-inf for the ~34 firm-months where 12-month-lagged annual at == 0; the engine yields NaN there. The reference is wrong: a tax change scaled by zero assets is undefined, and propagating inf into a cross-sectional signal is an unintended artifact, not methodology. Correct definition: NaN whenever the lagged-assets denominator is 0 or missing.

### ChangeInRecommendation
ChangeInRecommendation: engine takes each analyst's chronologically last recommendation in the month (sort by anndats before groupby.last), averages across analysts, computes opscore = 6 - mean and its 1-month change, then restricts to the SignalMasterTable permno-month universe via SMT's tickerIBES mapping. Deviation: the reference's groupby(...).last() is applied to UNSORTED data, so the analyst's 'last' rec within a month is whichever row happens to appear last in the parquet file — on multi-rec analyst-months this can select a chronologically earlier recommendation (proven: HAL 1999-03). The reference is wrong by its own stated definition ('last non-missing recommendation'); file order is not a temporal ordering. Correct definition: the recommendation with the latest anndats per analyst-firm-month. Affects ~55 cells (max abs diff 2 on the opscore-change scale).

### CompositeDebtIssuance
CompositeDebtIssuance: We compute log((dltt+dlc)_t/(dltt+dlc)_{t-60}) with a strict calendar 60-month lag. The OSAP script (CompositeDebtIssuance.py L70-71) falls back to a positional groupby.shift(60) when the exact calendar t-60 row is missing; on gapped Compustat panels this reaches back MORE than 60 calendar months, so the oracle emits '5-year debt growth' values actually measured over longer, unstated horizons (verified: 19,518 positional-fallback cells, missing-in-native only — 0 value divergence, inf cells matching sign-for-sign). The correct definition is a fixed 60-calendar-month window; where the t-60 observation does not exist the value is undefined and we emit NaN (P4). All other oracle differences were engine defects (missing dlc, spurious 12-month leg) and have been fixed to match the reference formula (div/log left unguarded: lag==0 -> +inf, debt==0 -> -inf, both kept like the oracle; raw m_aCompustat support, no SMT/span wrap).

### DebtIssuance
DebtIssuance: engine adopts the reference's universe conditioning (SMT-matched permno-months, shrcd<=11, BM=log(ceq/mve_permco) defined) as correct methodology, but deviates on missing-data handling: reference DebtIssuance.py:54 (and the OSAP Stata original) codes DebtIssuance=0 when dltis is missing, silently classifying ~358k firm-months with unreported debt issuance (largely pre-funds-statement-era Compustat) as non-issuers. The reference is wrong under NaN discipline: an unreported cash-flow item is unobserved, not zero. Correct definition: DebtIssuance = 1 if dltis>0, 0 if dltis reported and <=0, NaN if dltis unreported; restricted to SMT common stocks (shrcd<=11) with defined log book-to-market. Oracle cells whose only support is the NaN-to-0 fill are masked as oracle_artifact in reconciliation.

### DelCOA
DelCOA: engine computes ((act-che)_t - (act-che)_{t-12cal}) / (0.5*(at_t + at_{t-12cal})) with strict calendar lags; reference DelCOA.py:51-53 uses positional groupby.shift(12), so across m_aCompustat coverage gaps the lagged act/che/at come from 13 to 48+ months earlier (miss mask bit-identical to ChInv's). The reference is wrong: Richardson et al. (2005) Table 8C is a one-year change scaled by one-year-average assets; OSAP's Stata original used calendar l12. Correct definition: NaN whenever no observation exists exactly 12 calendar months prior. The 15,586 across-gap oracle cells are masked as oracle_artifact.

### DolVol
DolVol: WITHDRAWN as a deviation (2026-07-31) — superseded. The engine previously emitted NaN where the lagged dollar volume is zero (lagged vol==0); it now leaves the log unguarded and emits -inf there, bit-identical to the oracle's 10,079 np.log(0) = -inf sentinel cells (DolVol.py L34, L38) — the frozen golden run shows 0 missing-in-native cells, confirming no -inf/NaN mismatch remains. All other differences were engine defects (missing log, missing 2-month lag, rolling-mean instead of point lag) and have been fixed to match the reference formula. Sole remaining residual: 30,177 engine-only cells where the calendar LOG(LAG(vol*|prc|,2)) on the wide grid emits into months with no monthlyCRSP row (the script's positional shift(2) cannot), joint values bit-identical — covered by the calendar-vs-positional family entry below (Accruals, ChEQ, DolVol, GrAdExp, AccrualsBM).

### DownRecomm
DownRecomm: We take each analyst's chronologically last recommendation within the month (sort by anndats, then last) before averaging across analysts. The OSAP script (DownRecomm.py L31-35) applies .last() per analyst-month in raw FILE order without sorting by announcement date, so when the IBES file stores a month's records out of chronological order the script selects an OLDER recommendation as the month's stance (verified DANB/permno 81695 2002-05: script picks the 2002-05-20 rec over the 2002-05-31 rec). The correct definition is the latest recommendation as of month end; 14 oracle cells differ and are script sort-order artifacts. Engine output is additionally masked to SignalMasterTable membership to match the script's intended universe filter.

### GrLTNOA
GrLTNOA — engine deviates from OSAP GrLTNOA.py L83-86. Reference computes 12-month lags of the 10 balance-sheet inputs via positional groupby.shift(12) over m_aCompustat rows; across coverage gaps the lag base month can be years-to-decades before t-12 (observed up to 30 years), so the reference values an 'annual growth in LTNOA' against an arbitrarily stale base. Correct definition: all lagged inputs are taken at calendar month t-12 exactly; if the firm has no m_aCompustat observation at t-12, GrLTNOA is NaN at t. Impact: 15,333 oracle-only cells are dropped as artifacts; zero extras; all shared cells match to 0 abs error.

### GrSaleToGrInv
GrSaleToGrInv: (a) Engine adopts the script's 0-denominator guard — primary = DIV0(sale-avg,avg) - DIV0(invt-avg,avg), fallback = DIV0(sale-l12,l12) - DIV0(invt-l12,l12), DIV0 returning NaN when the denominator is exactly 0, coalesced primary-then-fallback (reference-correct; prior engine inf behavior was a defect, now fixed). (b) DELIBERATE DEVIATION on lag semantics: the Python reference uses positional shift(12/24) over observed Compustat rows, so across coverage gaps its '12/24-month' lags reach arbitrarily far back (e.g. permno 18331 @1954-06 uses data >24 calendar months old and the oracle reports the primary formula where only the fallback is defined). The reference is wrong: the definition (Abarbanell-Bushee growth vs the average of values 12 and 24 months prior; OSAP Stata l12./l24. calendar operators) requires calendar lags. Correct definition retained by the engine: LAG(x,12)/LAG(x,24) on the calendar monthly grid; if the calendar month has no observation the term is NaN and the 12m fallback (itself calendar) applies. Measured residual: 12,568 oracle-only cells (no calendar t-12 row; engine NaN) + 9,759 value cells (max abs diff 142.5) where calendar t-24 is absent and the engine's calendar 12m fallback replaces the script's stale positional primary — 22,327 cells, ~0.9% of joint support (the previous ~0.4% figure was stale); 0 extra-in-native, 0 inf. Those oracle values use stale data and are documented as reference defects, not engine errors.

### GrSaleToGrOverhead
GrSaleToGrOverhead: (a) Engine adopts the script's 0-denominator guard — primary = DIV0(sale-avg_sale,avg_sale) - DIV0(xsga-avg_xsga,avg_xsga), fallback = DIV0 over 12m lags, DIV0(.,0)=NaN, coalesced primary-then-fallback (reference-correct; prior engine inf leakage was a defect, now fixed). (b) DELIBERATE DEVIATION on lag semantics: the Python reference's positional shift(12/24) over observed rows reaches beyond the stated 12/24-month window across Compustat coverage gaps, contradicting both the economic definition and the OSAP Stata calendar l12./l24. operators. Correct definition retained by the engine: calendar LAG(12)/LAG(24) on the monthly grid, NaN when the calendar month is unobserved, calendar 12m fallback. Final residual vs oracle (grand_sweep 2026-07-31): 10,718 both-valued gap cells beyond tol (0.41% of joint support, max abs 336.9) + 12,400 oracle-only support cells at calendar-gap lags; zero extras; all inf cells eliminated. These oracle diffs are documented reference defects.

### Herf
Herf: engine computes industry sales Herfindahl over the SignalMasterTable universe (shrcd in {10,11,12} and exchcd in {1,2,3}, listed that month) and broadcasts each industry-month value to every SMT member of the industry regardless of the firm's own sale being missing — both matching the reference. TWO DOCUMENTED DEVIATIONS where the reference is wrong: (a) the script's transform('sum') emits tempHerf = 0.0 for industry-months in which every member's sale is NaN (pandas sum of empty set = 0), injecting spurious zero-concentration observations into the 36-month rolling mean; correct definition: the industry Herfindahl is undefined (NaN) when no member has sale data, and the NaN is simply a skipped observation in the 36m mean (min 12 obs). (b) the script's regulated-industry exclusion list contains the typo " 4813" (leading space, Herf.py L86), so SIC 4813 firms are never excluded for years <=1982 in the oracle; the stated Barclay-Smith 1995 rule (and the sibling HerfAsset script) excludes 4812 AND 4813 pre-1983; the engine excludes both. Oracle cells present for SIC-4813 firm-months <=1982 are reference defects.

### InvestPPEInv
InvestPPEInv — engine deviates from OSAP InvestPPEInv.py L68-70. Reference computes l12_ppegt/l12_invt/l12_at via positional groupby.shift(12); across m_aCompustat gaps the 'one-year' change spans up to 22+ years. Correct definition: delta(ppegt,12), delta(invt,12) and lag(at,12) taken at calendar t-12 exactly; NaN when no t-12 observation exists. The reference's l12_at==0 -> NaN guard (L82-86) is adopted as correct (engine tree amended with ZERO_TO_NULL on the denominator, eliminating 30 spurious inf cells). Impact: 14,319 oracle-only gap-lag cells dropped as artifacts; shared cells match exactly.

### Investment
Investment: NARROWED (2026-07-31). The engine now deliberately replicates the reference's polars inf-PROPAGATING rolling mean: an inf capx/revt ratio (revt==0 month) inside the trailing 36m window makes the historical mean ±inf (mixed signs -> NaN), and value = ratio/inf emits the oracle's exact-0.0 blocks; the observation gate counts inf ratios as observations. ONE residual deviation kept: on all-zero-ratio windows polars emits ±0.0 from ~1e-17 float noise where the true math is 0/0; the engine keeps NaN there (~194 oracle-only cells, e.g. permno 12072 — registered oracle artifact). Additionally ~25 both-valued cells fail the 1e-6 absolute tolerance purely from float32 storage of |value|~30-2500 (relative agreement ~1e-8). Zero extra-in-engine cells.

### NOA
NOA: engine computes (OA-OL)/at_{t-12cal} with a strict calendar lag; reference NOA.py:56 uses positional groupby.shift(12), so across m_aCompustat coverage gaps the denominator is total assets from up to 12 years earlier (e.g. permno 81021 @ 2019-08 scaled by at from 2007-08). The reference is wrong: Hirshleifer et al. (2004) Table 4 scales by one-year-lagged total assets; OSAP's Stata original used calendar l12. Correct definition: NOA_t = ((at-che) - (at-dltt-mib-dc-ceq))_t / at_{t-12cal}, NaN when no observation exists exactly 12 calendar months prior. The 15,351 across-gap oracle cells are masked as oracle_artifact.

### NetPayoutYield
NetPayoutYield — engine matches all reference filters (non-financial SIC, ceq>0-or-missing, SignalMasterTable universe, SMT-based t-6 mve_permco lag, >=24 prior eligible observations with the count taken over post-zero/SIC/ceq-filter rows) but deviates from OSAP NetPayoutYield.py L62-68: where (dvc+prstkc-sstk)/mve_l6 nets to exactly 0 with at least one nonzero component, the reference stores the sentinel 1e-19; the engine stores the true value 0.0. Reason: 1e-19 is a fabricated constant used only to defeat the script's own zero-drop filter; the economically defined value is 0. Impact: 828 cells valued 0.0 in engine vs 1e-19 in oracle; observations where dvc=prstkc=sstk=0 remain excluded in both.

### NumEarnIncrease
NumEarnIncrease: RESOLVED as deliberately matched (2026-07-31, no longer a deviation). The engine now rebuilds the script's exact merged panel — SignalMasterTable rows with a gvkey inner-joined to m_QCompustat ibq — and evaluates chearn = ibq - ibq(t-12cal) and every 3..24-month calendar lag ON that U-masked panel, so a (permno, t-k) pair absent from the merged panel lags to NaN exactly like the script's self-merge (rows present with NaN ibq stay in the panel but carry NaN), and output support is the merged panel itself. The previous stance (compute lags from the FULL quarterly history so universe gaps don't destroy real observations) is withdrawn in favor of exact reconciliation; the ~5,000 formerly-diverging boundary cells now match the oracle.

### OrderBacklog
OrderBacklog — engine deviates from OSAP OrderBacklog.py L33. Reference lags at via positional groupby.shift(12), so across m_aCompustat gaps the 'average total assets' denominator mixes at(t) with at from a month up to 8+ years before t-12. Correct definition: OrderBacklog = ob / (0.5*(at(t) + at(t-12 calendar))), NaN when ob=0 or no observation exists at calendar t-12. Impact: 3,033 oracle-only gap-lag cells dropped as artifacts; zero extras; all shared cells match to 0 abs error.

### OrderBacklogChg
OrderBacklogChg: we use true calendar 12-month lags for both at_lag12 and OrderBacklog_lag12. The reference Python script uses positional groupby.shift(12), which at panel gaps reaches >12 calendar months back and produces values where the calendar lag is missing (4 value cells beyond tolerance + 4,690 coverage-only cells). The original Stata implementation (xtset + l12.) is calendar-true, so the Python oracle deviates from its own published definition; our engine matches the Stata/published definition. Correct definition: OrderBacklogChg_t = OB_t/avg(at_t, at_{t-12}) - OB_{t-12}/avg(at_{t-12}, at_{t-24}) with all lags at exact calendar offsets, NaN when any calendar lag is unobserved.

### PS
PS (Piotroski F-score): RESOLVED as deliberately matched (2026-07-31, no longer a deviation). The engine now masks EVERY input to the script's merged-panel universe U (m_aCompustat ∩ SignalMasterTable ∩ monthlyCRSP row presence) BEFORE taking the 12-calendar-month lags, reproducing fill_date_gaps + stata_multi_lag exactly: a lag lands NaN when (permno, t-12) is not in U or the variable was NaN there, and the Stata missing=+inf convention then fires — including on p9, where a missing merged-panel lag row nulls a shrout value that exists in CRSP and awards p9=1 (the previously registered ~14,000-cell deviation is withdrawn for exact reconciliation). BM top-quintile now uses per-month pd.qcut value breakpoints on log(ceq>0 / mve) with duplicates='drop' (a collapsed top label < 5 NaNs the whole month, like the script); output masked to U. Only theoretical residual: duplicate (permno, month) m_aCompustat rows (engine pivot keeps 'last' vs the script keeping duplicate merged rows) — currently zero such rows.

### REV6
REV6: base panel, price source, and universe follow the reference (SignalMasterTable rows and prc, all (tickerIBES, permno) pairs, output restricted to SMT), but all lags — meanest_lag1, prc_lag1, and the six tempRev lags — are true calendar-month lags, missing when the prior calendar month is unobserved. The reference Python script uses positional groupby.shift over SMT row order, so at listing gaps it treats a many-months-old estimate/price as the 1-month lag (permno 62519/BLK 1995-05: divides by a price from 1994-05, 12 months stale, REV6 0.16025 vs 0.0030 with true lags). The original Stata l. operator is calendar-true and would yield missing there, so the oracle deviates from its own published definition. Correct definition: tempRev_t = (meanest_t - meanest_{t-1})/|prc_{t-1}| at exact calendar offsets; REV6_t = sum of tempRev_{t-5..t} plus current per the 7-term window, NaN if any calendar term is missing. Measured residual after the calendar-lag patch: 711 missing-in-native cells (oracle-only), all at permno-months whose 7-term lag window (t-7..t) spans an SMT listing gap, where the reference's positional shift emits a value from stale meanest/prc and the engine's calendar lags correctly yield NaN; joint-support values agree to <1.2e-10; zero extras. The previously cited 91 cells were the pre-patch both-valued divergences of the same family.

### SurpriseRD
SurpriseRD — deviates from OSAP pyCode SurpriseRD.py on the 12-month lag semantics. The script uses positional groupby(permno).shift(12) (L33-34), which across m_aCompustat coverage gaps compares current xrd against an observation more than 12 calendar months old (stale base for the surprise conditions xrd/xrd_lag12 > 1.05 and (xrd/at)/(xrd_lag12/at_lag12) > 1.05), emitting ~6.7k firm-months where no true t-12 observation exists. The original Stata implementation used calendar l12. lags. The engine uses calendar t-12 lags on the monthly wide grid and emits NaN when the permno has no m_aCompustat row at t-12 (condition6 fails). Correct definition: SurpriseRD_t = 1 if xrd_t/revt_t>0, xrd_t/at_t>0, xrd_t/xrd_{t-12}>1.05, and (xrd_t/at_t)/(xrd_{t-12}/at_{t-12})>1.05, with both xrd_t and xrd_{t-12} observed at calendar t-12; 0 if both observed but conditions fail; NaN otherwise. Zero value disagreements exist where both implementations emit a value.

### TotalAccruals
TotalAccruals — deviates from OSAP pyCode TotalAccruals.py in three ways. (1) Late (post-1989) branch: engine uses zero-filled sstk/prstkc/dv (late = ni - (oancf+ivncf+fincf) + (sstk0 - prstkc0 - dv0)); the script creates these zero-filled temps at L53-55 but then accidentally uses the raw columns at L88-92, so any missing supplementary item NaNs the whole signal and drops the firm-month. The zero-fill is the documented Richardson et al. (2005) treatment and matches the script's own declared intent; the raw-variable use is a pyCode translation bug (~22.3k firm-months silently lost). (2) 12-month lag: engine uses calendar t-12 on the monthly wide grid; the script's positional groupby.shift(12) reaches arbitrarily stale rows across m_aCompustat coverage gaps (observed lags to 30+ months), violating the 12-month definition; the original Stata l12. is calendar. Engine emits NaN when no calendar t-12 row exists (~14.9k oracle cells not reproduced, by design). (3) at_lag12==0: engine returns NaN (undefined scaling); the script emits inf which survives its dropna (108 oracle cells). Correct definition: TotalAccruals_t = [dWC + dNC + dFI]/at_{t-12} for year<=1989 and [ni - (oancf+ivncf+fincf) + (sstk0-prstkc0-dv0)]/at_{t-12} for year>1989, all lags calendar 12-month, result NaN when at_{t-12} is missing or zero.

### UpRecomm
UpRecomm: we select each analyst's last recommendation of the month by announcement date (anndats ascending, take last). The reference script calls groupby(...).last() on the raw parquet row order with no date sort, so its within-month 'last' depends on arbitrary file ordering and flips the signal where an analyst revised twice in a month (8 cells, e.g. MRBA 1999-05: anndats 5/25 ireccd 2.0 then 5/27 ireccd 3.0; file order keeps 2.0, chronology keeps 3.0). Correct definition: the analyst's month-end stance is the chronologically latest recommendation by anndats. Universe and multi-permno fan-out follow the reference (all SMT (tickerIBES, permno) pairs, output masked to SignalMasterTable).

### grcapx
grcapx (updated 2026-07-31): ONE deviation remains — calendar lags. l24.capx (and the fallback's l12.ppent) mean the value at time_avail_m minus 24 (12) calendar months, NaN if that month is absent from the SMT-masked panel; the script uses positional groupby.shift on the SMT-inner-merged panel, so across universe-exit gaps its 'l24' silently reaches >24 months back (e.g. permno 36767 1974-07). The Stata source (tsset + l24.) is calendar-based, so the Python script is unfaithful to its own reference — order ~10k oracle-only/value cells at gap permnos, including inf cells landing at different months. Withdrawn former items: FirmAge for the capx-patch gate is now the script's cumulative SMT row count (censored via tempcrsptime — see the FirmAge kernel), and l24_capx==0 now KEEPS inf exactly like the script (both sides emit inf; only the month placement can differ via the lag deviation). Adopted as reference methodology: SMT universe (smt_only inputs + SMT_OUT support) and the missing-capx := ppent - l12_ppent patch gated on FirmAge >= 24 months.

### grcapx3y
grcapx3y: Same register entry as grcapx items (a)-(c) — calendar lags at 12/24/36 months instead of the script's positional shifts on the gapped SMT panel; calendar FirmAge for the capx patch gate; and additionally the denominator (l12_capx + l24_capx + l36_capx) == 0 yields NaN, whereas both the script (2,332 cells) and the previous engine (3,016 cells) emitted inf. SMT universe mask and the ppent capx patch are adopted as the reference's deliberate methodology.

### std_turn
std_turn — engine deviates from OSAP std_turn.py L47-52. Reference computes rolling_std of turnover (vol/shrout) over the trailing 36 ROWS per permno with min 24 samples; for permnos with listing gaps this window spans more than 36 calendar months, mixing pre- and post-gap turnover regimes while labeling the result 'past 36 months volatility'. Correct definition: standard deviation of monthly turnover over calendar months t-35..t, requiring >=24 non-null months, NaN otherwise. The reference's size screen is adopted as correct and matched exactly: size = shrout*|prc| from monthlyCRSP, per-month pd.qcut-equivalent quintile breakpoints, values kept for quintiles 1-3, null when quintile>=4 or size is missing (spec fix: min_tercile null so no implicit bottom-tercile exclusion). Impact: cells for gap-spanning windows differ from oracle by construction; boundary-quintile extras eliminated by qcut-equivalent breakpoints.

### LRreversal
Engine compounds returns over calendar months t-13..t-36 (lag on the month grid, NaN
when a lag month precedes the listing span; in-span gap months contribute zero return).
Reference script LRreversal.py shifts POSITIONALLY over the un-gap-filled SignalMasterTable
rows (`groupby(permno).shift(i)`), so for firms with coverage gaps its "24-month window"
silently spans more than 24 calendar months of history. The reference is wrong: De Bondt-
Thaler long-run reversal is defined over a fixed calendar horizon. Measured residual vs
oracle after all fixes (2026-07-31): 36,258 missing-in-engine cells (SMT gap inside the
t-13..t-36 window), 3,915 both-valued cells beyond tol (max abs 26.8), and 140
extra-in-engine cells (>=36 calendar months but <36 SMT rows) — all the same
positional-shift-over-gappy-frame deviation, accepted as registered.

### AnalystRevision
Engine: FY1 consensus (fpi='1') at month t divided by the FY1 consensus at calendar
month t-1, on SignalMasterTable rows. Reference script lags POSITIONALLY over SMT rows,
so after a coverage gap it divides by a consensus more than one month old. The reference
is wrong: the revision ratio is defined against the prior month. Measured residual after
the fpi fix: 105 differing cells + 7/106 support cells (0.006% of joint support), every
sampled one immediately following an SMT coverage gap.

### BM (book-to-market)
Engine: market equity frozen at the firm's latest fiscal-year-end month (datadate-matched,
carried forward), per the reference's own intent. Reference script matches the datadate
month via POSITIONAL shift(6) over SMT rows, so firms with listing gaps freeze the wrong
month's market equity. Measured residual after implementing the true freeze semantics:
1,436 differing cells + 1,036 support cells (0.09% of joint support), gappy firms only.

### Accruals, ChEQ, DolVol, GrAdExp, AccrualsBM — calendar vs positional lags (residual support)
Same deviation family as LRreversal/AnalystRevision/BM: these scripts lag by ROW
POSITION over gappy frames, so across coverage gaps they pair month t with data older
than the stated horizon (or produce values where the true t-k month has none). The
engine uses true calendar lags. Measured residuals after the formula fixes (values
bit-identical on joint support): Accruals 15,359 missing-in-engine cells; ChEQ 11,817;
DolVol 30,177 engine-only cells (raw-CRSP universe, no SMT filter in its script);
GrAdExp 2,739 missing-in-engine cells (oracle-only) — the script's positional
shift(12) over gappy m_aCompustat supplies a stale xad base (sampled up to 19 years
old, e.g. permno 10517 2011-06 lagged to 1992-09) where no calendar t-12 row exists,
engine correctly NaN; 0 extras, 0 value diffs (the previously predicted 2,404 count
was a pre-support-fix simulation).
AccrualsBM — calendar vs positional 12-month tempacc lags over the SMT right-join
panel. Engine computes working-capital accruals with calendar t-12 lags on the
m_aCompustat grid, then buckets BM and accruals into per-month fastxtile quintiles
over SMT rows; reference AccrualsBM.py shifts positionally over the SMT right-join
panel, so across listing/coverage gaps its tempacc uses stale lags — producing both
direct residuals (oracle values where no calendar t-12 exists) and indirect fastxtile
breakpoint flips on boundary cells whose own inputs are identical. Measured residual:
532 missing-in-native + 740 extra-in-native, 0 value differences on joint support.

### ChTax — gvkey-keyed vs permno-keyed 12-month lags (residual)
CORRECTED MECHANISM (the previous 'calendar vs positional' description was wrong —
the script's lags ARE calendar-based): the reference lags txtq/at by GVKEY, so a
company keeps its lag continuity across permno changes; the engine's wide-grid lags
are keyed by PERMNO, so months within 12 months of a gvkey-link change lose (or gain)
the lag observation. Residual after the denominator fix: 175 differing cells (max
~0.0328) + 4,867 missing-in-native / 1,824 extra-in-native support cells (~0.006% of
support), all gvkey-link-churn cells. Exact reconciliation would require a
gvkey-keyed lag operator, not a positional shift.

### ExchSwitch — calendar vs positional 12-month exchcd lookback (residual)
Engine flags a switch onto NYSE/AMEX when any of the 12 CALENDAR months t-1..t-12
carries the prior exchange code (exchcd SMT-masked, NaN lags compare False; 0/1 on
SMT rows). The reference's shift(1..12) runs positionally over gappy SMT rows, so for
permnos with listing gaps (delist -> gap -> relist, exactly where switches occur) it
can reach exchcd observations MORE than 12 calendar months back. One-directional
residual: native=0 vs oracle=1 on a small number of post-gap relisting cells; the
engine's fixed 12-calendar-month window is the stated definition.

### AnalystRevision (final) — calendar vs positional 1-month lag
After masking the lag input to SignalMasterTable rows, engine values are bit-identical
to the reference everywhere both emit. Residual: 127 cells where the reference's
positional shift(1) reaches across a listing gap to a months-old consensus (proven:
permno 17347 1982-08 divides by the May consensus across a Jun-Jul gap); the engine's
calendar lag correctly yields NaN there.

### DelCOL
DelCOL: engine computes ((lct-dlc)_t - (lct-dlc)_{t-12cal}) / (0.5*(at_t + at_{t-12cal})) with strict calendar lags; reference DelCOL.py:64-66 uses positional groupby.shift(12) on deduped m_aCompustat rows, so across coverage gaps the lagged lct/dlc/at come from 13 months to 14+ years earlier (e.g. permno 89986 2023-03 differenced against 2009-03; miss mask matches Accruals' 15,359-cell coverage footprint). The reference is wrong: Richardson et al. (2005) defines a one-year change scaled by one-year-average assets; OSAP's Stata original used calendar l12. Correct definition: NaN whenever no observation exists exactly 12 calendar months prior. The 15,359 across-gap oracle cells are masked as oracle_artifact.

### DelDRC — calendar vs positional 12-month lag (residual support)
Same family as DelCOA/ChInv: reference DelDRC.py:70-71 lags drc/at via positional groupby.shift(12) over deduped m_aCompustat rows, so across coverage gaps the 'annual' change in deferred revenue is differenced against data 13 months to 12+ years old (sampled: permno 81575 2011-05 vs 1999-05). Engine uses strict calendar t-12 lags and emits NaN when no m_aCompustat observation exists exactly 12 calendar months prior. The 1,884 across-gap oracle cells are masked as oracle_artifact.

### NetDebtFinance — calendar vs positional 12-month at lag (residual)
Engine computes (dltis - dltr + dlcch_fillna0) / avg(at, at_{t-12cal}) with |x| > 1 -> NaN (the script's trim) on the raw m_aCompustat universe — formula, trim, and dlcch sign reference-exact. Residual: the reference's positional groupby.shift(12) for l12_at (NetDebtFinance.py L45) bridges m_aCompustat calendar gaps (e.g. permno 39888 1972-09 scaled by at from 1970-06), emitting 13,727 oracle-only cells where the calendar t-12 row does not exist; engine NaN there is P4-correct. 0 value-divergent and 0 extra cells. Same family as NetEquityFinance/OrderBacklogChg.

### NetEquityFinance — calendar vs positional 12-month at lag (residual)
Engine computes (sstk - prstkc - dv) / avg(at, at_{t-12cal}) with |x| > 1 -> NaN (the script's trim) on the raw m_aCompustat universe. The reference's positional groupby.shift(12) bridges m_aCompustat calendar gaps to horizons longer than 12 months, so the oracle emits exactly 13,542 cells where the true calendar t-12 at does not exist (engine NaN, P4-correct); the frozen golden run shows 0 value-divergent and 0 extra cells — the formerly anticipated small sliver of value cells did not materialize. Same family as Accruals/AssetGrowth; the formula itself (dv term, trim, denominator) is now reference-exact.

### PctAcc — calendar vs positional 12-month lags in the accrual fallback (residual)
Engine numerator = ib - oancf, falling back to (dACT-dCHE) - (dLCT-dDLC-dTXP-dp) with strict calendar 12-month deltas only when oancf is missing; denominator = |ib| with 0.01 substituted ONLY at ib==0 and NaN when ib is NaN — all reference-exact. Residual: exactly 3,870 oracle-only cells — engine NaN at true-calendar-gap months where the reference's positional shift(12) of act/che/lct/dlc/txp bridges m_aCompustat coverage gaps; the frozen golden run shows 0 value-divergent and 0 extra cells. Same registered family as Accruals.

### ShareVol — calendar vs positional 1/2-month lags at SMT gaps (residual)
Engine computes 3-month average turnover and the share-count-change exclusion with calendar lags over SMT-masked vol/shrout; missing turnover emits 1.0 (the script's null branch) and obs_num counts SMT rows. Residual: at SMT listing gaps the oracle's positional shift(1)/shift(2) compares against pre-gap vol/shrout while the calendar lag sees NaN — post-gap months get engine 1.0 (NaN turnover) vs the oracle's gap-jumping numeric turnover, and post-gap dshrout False vs the oracle's pre-gap comparison. Measured residual: 464 both-valued flips (native 1.0 from calendar-NaN turnover vs oracle 0.0 from gap-jumping positional turnover) plus 2,864 native-only cells (the oracle drops post-gap rows because positional dshrout/l1/l2_dshrout compare shrout across the gap and trip dropObs; engine calendar comparisons are False and keep the SMT row); all at SMT listing gaps, everything else exact (replaces the stale order-tens estimate).

### MS — polars/pandas float knife-edges on indicator flips (oracle artifact)
Support is exact (SMT∩Compustat∩quarterly universe, fiscal-timing mask incl. the reference's June-FYE bug, Stata missing=+inf medians, month-gap-aware rolling stats). Residual: ~3,503 both-valued cells (0.74%, 227 permnos, |diff| 1-2 score points) where an input sits exactly at its industry median or a rolling stat differs in the last float bits between polars and pandas arithmetic, flipping a 0/1 indicator that is then ffilled up to 12 months. Bit-level arithmetic noise, not methodology — registered oracle artifact.

### VolSD — file-order rolling windows in the reference (oracle artifact)
Engine computes the 36-month rolling std (min 24 obs) over CALENDAR/chronological monthly windows on the raw-monthlyCRSP row support (crsp_only). The reference script never sorts, so its rolling windows run over raw parquet ROW ORDER; for the 505 permnos whose monthlyCRSP rows are not stored chronologically the oracle's 'trailing 36 months' mixes arbitrary months. Measured residual (2026-07-31): 14,283 value cells beyond 1e-6 tolerance (max ~30.2) plus 801 missing-in-native and 900 extra-in-native support cells, across 421 divergent permnos — all confined to the 505 non-chronological permnos (verified exhaustively; the previously cited 359-permno figure and ~800-support-cell total were stale). Script defect: a rolling window over unordered rows is not a temporal statistic.

### VolMkt — file-order rolling windows in the reference (oracle artifact)
Same mechanism as VolSD: the reference script never sorts, so its trailing 12-row window over vol*|prc|/mve_c runs in raw parquet ROW ORDER; for the 505 permnos whose monthlyCRSP rows are not stored chronologically the oracle mixes arbitrary months. Engine uses chronological windows. Measured residual (2026-07-31): 4,507 value cells beyond 1e-6 (max ~5.26) plus 840 missing-in-native and 928 extra-in-native support cells, spread over 432 permnos, ALL confined to the 505 non-chrono permnos (verified exhaustively); everything off those permnos is exact.

### RevenueSurprise — panel-value gvkey mapping vs SMT gvkey (residual)
Engine masks revtq/cshprq to SMT-rows-with-gvkey before the seasonal z-score, so every calendar lag/drift/SD term sees the script's SMT-gvkey ∩ m_QCompustat row set, and keeps only SD > 1e-8 (script L99). Residual: rare permno-months where the m_aCompustat first-gvkey mapping that built the wide panel VALUES disagrees with SMT's gvkey for that permno-month (a different firm's revtq/cshprq enters the engine panel). Exact reconciliation would require building values through SMT's own gvkey link per month.

### SMT universe emit-mask family (AM, AdExp)
Engine kernels compute the ratio wherever the m_aCompustat and monthlyCRSP inputs exist; the reference scripts inner-join SignalMasterTable, so oracle support additionally requires SMT membership (shrcd 10/11/12, exchcd 1/2/3). Golden native outputs are deliberately frozen pre-mask; p1_universe_policy.json applies smt_strict at emit time for exact parity. Residuals: AM 281,901 extra-in-native off-SMT cells (joint values bit-identical); AdExp 62,000 extra-in-native off-SMT cells (joint values identical to float32 precision, max abs diff 7.3e-12); missing=0 for both.

### BMdec
BMdec: engine computes BE(t)/DecME(prior calendar year) with December market equity taken from the CRSP December row and lagged 12/17 calendar months, output masked to monthlyCRSP row presence. Reference BMdec.py fetches DecME by merging on a t-12/t-17 row of the m_aCompustat x CRSP inner panel and computes tempDecME only on merged rows, so it drops firm-months where the December ME exists in CRSP but the firm lacks a merged row at exactly the lag month or lacks Compustat coverage in the lag-year December. The reference is wrong: Fama-French Ln(BE/ME) requires only current book equity and the prior December price/shares; the merged-row-presence requirement is a lookup-implementation artifact. Measured residual: 318,101 engine-only cells (all with a CRSP row at t) retained by design; missing=0; values bit-identical wherever both sides emit.

### DivYieldST — calendar vs filtered-positional lags (residual)
Engine computes div12 as a calendar rolling(12, min 1) sum of 0-filled SMT-masked dividends and the Ediv1 lags (shift 2/5/11 by cd3 payment frequency) on the calendar month grid; reference DivYieldST.py L81-102 computes rolling(12) positionally over SMT rows and shift(2/5/11) positionally over the div12>0-FILTERED rows. Consequences: (a) 24,624 engine-only cells, all valued 0.0, at the first eligible months of each dividend run — the calendar lag sees a genuine no-dividend month (Edy1=0 -> DivYieldST=0) where the script's filtered-row shift has no prior filtered row and drops the observation; (b) 420 oracle-only cells at eligibility gaps where the script's filtered-row shift bridges to pre-gap dividends the calendar lag correctly treats as absent; (c) 75 both-valued cells (max 3): qcut-tercile boundary flips (3v2/2v1) and 0-vs-tercile flips at the Edy1==0 boundary induced by the same lag differences shifting the per-month ranking universe. All are the registered lag-semantics deviation; the patch-spec estimate of ~O(hundreds) was low.

### MeanRankRevGrowth — engine-only support at non-Compustat months (universe mask)
The engine evaluates the 5-term weighted average of calendar-lagged ranks (t-12..t-60) on the wide grid, so it emits values at 139,762 permno-months with no m_aCompustat row at t (the script's output frame is m_aCompustat itself; sampled 12/12 + 5/5 have no comp or SMT row at t). Joint-support values are bit-identical (0 beyond-tolerance, 0 missing). Handled by p1_universe_policy.json policy m_aCompustat_rows applied at emit time; the golden native snapshot predates the mask.

### roaq — merged-panel lag vs calendar t-3 lag (residual support)
Same family as NumEarnIncrease merged-panel semantics: reference roaq.py (L28, L37, L44-50) inner-joins SMT-with-gvkey rows to m_QCompustat and looks up atq_lag3 inside that merged panel, so when the permno was outside SMT at t-3 the script lags to null even though raw quarterly atq exists at calendar t-3. The engine keeps the true calendar t-3 atq from the full quarterly history, with output masked to SMT. Residual: exactly 1,917 engine-only cells (0 missing, 0 value diffs); sampled cells (permno 73198 1989-10, 67790 1992-10) confirmed in-SMT at t and out-of-SMT at t-3.

### EarnSupBig — calendar vs positional lags, industry-broadcast amplification
Engine computes the earnings-surprise zscore with calendar lags (the EarningsSurprise
definition) on SMT-masked inputs; reference EarnSupBig.py shifts positionally over its
merged panel, bridging coverage gaps. Because the statistic is a value-weighted BIG-firm
industry mean broadcast to every smaller member, each gap-affected big firm's difference
propagates to its whole industry-month: residual beyond_tol=1,608,622 cells (max 15.08),
missing=18, extra=0 — large in cells, single-family in mechanism. Two-sided proof: each
divergent big firm has merged-panel gaps in its 36-month lag horizon and a positional
replica reproduces the reference exactly.

### RDAbility — calendar fiscal-year windows + deterministic dedup
Engine regresses on valid pairs within the trailing 8 FISCAL YEARS; reference compacts
invalid pairs (polars null_policy='drop'), so its window bridges fyear gaps and spans >8
years. Residual: beyond_tol=9,721 (219 firms), missing=3,731 (112 firms, 111 with fyear
gaps); proven two-sided on permno 22752. Additionally 3 extra cells from the reference's
UNSTABLE sort at gvkey-switch overlaps (engine keeps the deterministic gvkey-ascending
row) — same class as the AbnormalAccruals dedup note.

### RealizedVol — pre-Fama-French months retained
Engine computes realized volatility (std of daily returns, ddof=1, >=15 valid days) for
every qualifying stock-month, including 1926-01..1926-06. The reference computes it inside
a shared script whose FF3 regression (for IdioVol3F) requires daily Fama-French factors,
which begin 1926-07 — so the reference drops six months of RealizedVol for a data
requirement belonging to a DIFFERENT factor. Measured: exactly 3,097 engine-only cells,
all in 1926-01..1926-06; joint values bit-identical.

### ReturnSkew3F — no skewness for perfect-fit months
When a stock-month's returns are exactly explained by the FF3 regression, the true
residuals are zero and their skewness is mathematically undefined (0/0). The reference
computes skew on the leftover floating-point dust (~1e-20) and publishes noise; the
engine emits NaN (residual std <= 1e-10 guard). Verified single-mechanism: 100% of all
31,924 divergent/missing/extra cells vs the oracle sit at residual std < 1e-10 (max
1.1e-19). IdioVol3F is unaffected (dust std ~ 0 matches within tolerance) and passes
exactly. Final on-disk footprint after the guard: 28,681 missing-in-engine cells
(values elsewhere identical to 5.7e-14), zero value differences, zero extras.

### IdioVolAHT — perfect-fit windows keep their true zero
When a stock's 252-day window is exactly explained by the market model (typically
non-trading stocks with constant returns), the true residual RMSE is 0.0 — a defined
value, unlike skewness's 0/0. The engine emits 0.0; the reference's one-pass SSE
computation produces negative floating dust, sqrt fails, and the month is silently
dropped. Measured: 23 engine-only cells, all with native value exactly 0.0; joint
values elsewhere identical to 2.1e-11.

### Beta — no sub-minimum estimates from the library's degenerate branch
The reference computes rolling CAPM betas via polars_ols (60 valid observations,
minimum 20). PROVEN empirically: for stocks whose TOTAL valid history never reaches 20
observations, the library falls into a degenerate branch that emits coefficients from
the 20th raw row onward — betas estimated from as few as 13 pairs, below the
definition's own minimum (permno 10508: emission at 1928-02 with 13 valid pairs;
stocks above the threshold gate correctly on the 20th valid observation). The engine
gates on valid pairs uniformly. Measured: 1,706 oracle-only cells, 20/20 sampled with
total-valid < 20; joint values bit-identical, zero extras.
BetaLiquidityPS shares the identical mechanism at its 36-observation minimum:
5,013 oracle-only cells, 20/20 sampled with total-valid < 36; values bit-identical
(max 9.3e-10). ResidualMomentum likewise: 180 oracle-only cells, 15/15 sampled from
stocks with total-valid < 36 (its stage-1 residual regressions inherit the same
degenerate branch); values identical to 7.5e-9, zero extras. BetaTailRisk likewise
at its 72-observation minimum: 9,988 oracle-only cells, 20/20 sampled with
total-valid < 72; values identical to 1.2e-7, zero extras. Its regression MECHANICS are thereby validated against the published
Pastor-Stambaugh series; the point-in-time series swap (user decision) follows as a
separate, additional registered deviation when it lands.

### BetaLiquidityPS — point-in-time liquidity series (user decision)
The engine regresses on a REBUILT Pastor-Stambaugh liquidity-innovation series in which
the value at month t uses only data through t (expanding-window estimation, 60-month
burn-in, frozen once computed) — Data/ps_innov_pit.parquet, construction documented in
p1_kern_ps_liquidity.py. The published series embeds full-sample estimation: its
historical values change when re-estimated, and the divergence concentrates in crisis
months (1987-11: PIT -0.034 vs full-sample -0.092). Chain of evidence: regression
mechanics validated EXACTLY against the published-series oracle before the swap (values
to 9.3e-10); series-construction replica correlates +0.897 with the published series
(the literature's known replication ceiling; requires the positive-volume screen the
2003 paper never disclosed — documented in the authors' 2019 follow-up); PIT vs
full-sample replica correlate +0.996. Measured factor-level footprint vs the
published-series oracle: values differ on ~100% of joint cells (correlation 0.874),
support smaller by 119,687 cells (PIT series begins 1967-09 after burn-in). This factor
intentionally matches NO published benchmark.

## B. Look-ahead corrections 
### GP, tang — point-in-time industry classification
Engine screens (non-financial for GP, manufacturing for tang) use the HISTORICAL
Compustat industry code (sich) per fiscal year, falling back to the header code only
where sich is missing (early years, where no better information exists). Reference
scripts use the header code as of the data download, retroactively reclassifying a
firm's entire past when its industry changes — look-ahead in universe selection.
Measured footprint: joint values bit-identical; support differs by 19,967/26,868 cells
(GP) and 38,118/70,320 cells (tang) — the firms whose classification changed.

### IntanBM, IntanCFP, IntanEP, IntanSP — point-in-time trimming
Engine trims the 60-month return input at 1%/99% quantiles computed WITHIN EACH
MONTH's cross-section. Reference (winsor2 by=None) computes cutoffs over the entire
1926-2024 panel, so month-t trimming depends on future returns. Kernel verified exact
in REFERENCE_MODE on 2026-07-31 with the current (post-19:25) kernel — 0 beyond-tol
cells and 0 support diffs vs the oracle at atol 1e-6 / rtol 1e-5 for all four factors
(max abs diff: IntanBM 6.0e-16, IntanCFP 9.3e-10, IntanEP 4.5e-13, IntanSP 0.0) — so
each factor's production diff vs the oracle is attributable to the per-month-trim
deviation alone (trim selection changes each month's regression sample, hence
residuals ~everywhere plus small support churn). Production residuals: IntanBM
1,714,687 beyond-tol + 12,771 missing / 13,604 extra (max 2.36); IntanCFP 1,864,967 +
15,981/15,315 (max 23.4); IntanEP 1,865,004 + 15,981/15,315 (max 21.3); IntanSP
1,860,586 + 15,949/15,296 (max 45.4). (The frozen Data/golden/refmode_check/
Intan*.parquet and refmode_baseline.csv predate the fixed kernel and await refreeze.)

### ExclExp — point-in-time clipping
Engine clips at 1%/99% within each month's cross-section; reference clips at
full-sample quantiles (look-ahead). Fiscal-quarter alignment resolved (0 missing / 0
extra support cells). Sole residual: 44,955 cells (2.5% of 1,772,217 joint cells, max
abs diff 7.40) where the reference clips at full-sample bounds (-1.81, 2.70) and the
engine clips at each month's 1%/99% — the registered look-ahead deviation; interior
cells match exactly.

### AbnormalAccruals — prior-year model estimation
Engine estimates the Jones-model coefficients and trim bounds on the PREVIOUS fiscal
year's (industry) cross-section — fully published before any current-year statement is
stamped — and scores each firm with its own statement only. Reference estimates within
the SAME fiscal year, using peers' statements published up to ~11 months after the
firm's stamp (look-ahead). Production output therefore deliberately differs
~everywhere from the look-ahead oracle: 2,459,521 beyond-tol / 148,567 missing /
15,419 extra cells. REFERENCE_MODE parity verified on disk 2026-07-25 after the kernel patch: 17 value + 20
support cells out of 2.6M — all traced to the reference's own NONDETERMINISTIC sort on
duplicate (permno, fyear) keys (measured: ties resolve arbitrarily run-to-run in the
reference; the engine uses a deterministic stable sort). Everything reproducible-in-
principle matches exactly.

### VolumeTrend — full-sample trim now reproduced (2026-07-31)
SUPERSEDED: the engine previously applied no trim (the clean point-in-time choice) and
registered the reference's full-sample 1%/99% trim here as a look-ahead correction.
Per the verified kernel-patch worklist the spec now wraps VolumeTrend in trim_global
(nearest-interpolation full-sample quantile bounds, matching winsor2 by=None + polars
quantile default), DELIBERATELY reproducing the reference's look-ahead tail selection
to reconcile with the oracle. This is an accepted, documented look-ahead: whether a
month-t slope survives depends on the full-sample distribution. Swap trim_global for
_trim_by_month if point-in-time policy is later preferred (re-registers a ~2% boundary
residual).

(reference uses future information; we do not)

### AbnormalAccruals [lookahead_confirmed]
BOTH sides leak. Script ZZ2_AbnormalAccruals_AbnormalAccrualsPercent.py: winsor2 trim by=['fyear'] (line 138) and OLS residuals .over(['fyear','sic2']) (lines 151-163) pool the full fiscal-year cross-section, so a firm's residual stamped at its own datadate+6 depends on regression coefficients and trim cutoffs estimated from peers' statements published up to ~11 months later. Engine p1_factor_engine.py: _trim_by_group(..., 'fyear', 0.001, 0.999) (line 1690) and _ols_residuals_by_groups(group_cols=['fyear','sic2']) (lines 1692-1698) reproduce the same contemporaneous-fyear pooling, then _expand_hold_months holds the contaminated value 12 more months. (This mirrors Xie 2001's in-sample cross-sectional estimation, but it is a genuine availability violation at the stamped month.)

### BetaLiquidityPS [suspect]
Leak is in the input series, identical on both sides (engine is script-backed): LiquidityFactor.py stamps ff.liq_ps rows at their own month with no real-time vintage, and BetaLiquidityPS.py's trailing 60-month regression (lines 86-99, itself correctly backward-looking) consumes ps_innov values whose construction embeds full-sample (post-tau) information. This is the standard series used in the literature, but under a strict point-in-time criterion historical ps_innov values are revised ex post. Script's own alignment logic contains no other leak. Side note: the unexecuted engine spec declares window 36/min 12 vs the script's 60/36 — a parameter mismatch, not a leak.

### ExclExp [lookahead_confirmed]
Both sides winsorize with full-sample quantiles: script computes q1/q99 over the entire 1976-present panel then clips every month (ExclExp.py:47-50); engine does the same (p1_factor_engine.py:2377-2381). The clip bounds applied to month t depend on the distribution of future months' ExclExp, so stamped values in the ~2% tails are functions of future data (also creates ties at the bounds that can move portfolio-boundary ranks). Core signal inputs are timing-clean; the leak is in the transformation. (Separate non-timing note: availability-month merge can pair int0a and epspiq from different fiscal quarters — noise, faithful to OSAP.)

### GP [lookahead_confirmed]
The non-financial screen (SIC<6000 or >=7000) at every historical month t uses the firm's present-day header SIC on both sides (script GP.py:37-40; engine nonfinancial_only on the same m_aCompustat 'sic', p1_factor_engine.py:803-806). A firm that later became (or stopped being) a financial reclassifies its whole past, retroactively changing universe membership. The GP ratio itself is timing-clean; the leak is confined to the screen. Historical sich is downloaded (CompustatAnnual.py:37) but unused — the natural fix. Faithful to OSAP's Stata original, but a genuine leak under the absolute standard.

### IntanBM [lookahead_confirmed]
Both sides. tempRet60/ret60 is trimmed at 1%/99% quantiles POOLED OVER THE ENTIRE SAMPLE, so the trim threshold at month t depends on future months' 60-month returns. Script: ZZ1_IntanBM_IntanSP_IntanCFP_IntanEP.py:96 winsor2(..., by=None) -> utils/winsor2.py:217-237 (quantile over full panel). Engine: p1_factor_engine.py:1542 _op_intangible_residual calls _trim_global (lines 1210-1219, np.nanquantile over the whole month x permno panel). tempRet60 is both the regression y and part of the vRet regressor, so future data changes which obs enter the month-t cross-sectional regression and its fitted betas, i.e. the stamped residual. Second-order in magnitude (inherited from the OSAP Stata original) but a genuine full-sample statistic used in-sample.

### IntanCFP [lookahead_confirmed]
Same shared mechanism as IntanBM: full-sample 1%/99% trim of tempRet60/ret60 on both sides (script ZZ1 line 96 via winsor2 with by=None; engine _op_intangible_residual p1_factor_engine.py:1542 -> _trim_global lines 1210-1219). Trim cutoffs at month t are computed from all months including the future, altering the month-t regression sample/inputs and hence the residual factor value.

### IntanEP [lookahead_confirmed]
Same shared mechanism: pooled full-sample 1%/99% trim of the 60-month return on both script (ZZ1 line 96, winsor2 by=None) and engine (_op_intangible_residual line 1542 -> _trim_global lines 1210-1219). Future months' return realizations move the trim thresholds applied at month t.

### IntanSP [lookahead_confirmed]
Same shared mechanism: pooled full-sample 1%/99% trim of tempRet60/ret60 on both sides (script winsor2 by=None at ZZ1 line 96; engine _trim_global via _op_intangible_residual lines 1541-1542). Month-t trim cutoffs embed future-sample information.

### VolumeTrend [lookahead_confirmed]
Script side: VolumeTrend.py:79 calls winsor2(df, ['VolumeTrend'], replace=True, trim=True, cuts=[1,99]) with by=None; winsor2.py:122-138/162-170 then computes pl.col(var).quantile(0.01/0.99) over the full dataset and nulls observations outside those full-sample bounds — so whether a month-t value survives depends on future data (selection-only leak; surviving values are unchanged; inherited from the original Stata 'winsor2 ... trim' without by()). Engine side UPDATE (2026-07-31): the spec now wraps the slope in trim_global, deliberately reproducing the same full-sample trim for oracle reconciliation — the leak now exists on BOTH sides by documented choice (see 'VolumeTrend — full-sample trim now reproduced' above).

### realestate [lookahead_confirmed]
Engine side only: _op_industry_adjusted_mean (p1_factor_engine.py:887-900) forms 2-digit industry demeaning groups at month t from the classification as of the data-download date, so a firm reclassified in, say, 2015 is demeaned against its 2015 industry peers in 1990 — future classification info leaks into historical group assignment and the >=5-obs industry screen. Script side is clean (point-in-time sicCRSP). Fix: have the engine use sicCRSP (or funda's sich, already downloaded but unused).

### tang [lookahead_confirmed]
Both script (tang.py:35-38 filters on m_aCompustat sic) and engine (_op_manufacturing_only on the m_aCompustat sic panel, p1_factor_engine.py:809-812) decide historical sample membership using the download-date classification: a firm that became a manufacturer in 2010 is included back to the 1970s, and one that left manufacturing is excluded from years when it genuinely qualified. The leak is in universe selection, not the computed value; same-month at-decile FC flags in the script are constructed but never applied. Fix: gate on sich or CRSP's point-in-time sicCRSP.

## D. Data refresh — WRDS 2026-08-17 (history through 2024-12 preserved, 2025 appended)

Snapshot of the previous Intermediate/ kept at `pyData/Intermediate_2024snapshot/`; every
refreshed file was diffed cell-by-cell against it on the overlap by `p1_refresh_gate.py`
(reports: `Data/golden/reports/refresh_gate_20260817_*.json`). User policy: Compustat/IBES
restatements ARE accepted ("revised" history). Findings:

### CRSP — CIZ-format append (msf_v2 / dsf_v2), 2025-01..2025-12
CRSP retired the legacy stock-file format at 2024-12-31; WRDS `crsp.msf`/`crsp.dsf` end there,
`crsp.msf_v2`/`crsp.dsf_v2` continue to 2025-12-31. `DataDownloads/CRSPv2Append.py` keeps the
legacy download (bit-identical to the snapshot on all 5,153,763 monthly rows / 16 numeric
columns and on a 60-date daily sample: 0 changed cells) as the source of truth through 2024-12
and appends v2 rows after that. Mappings verified on the 2024-06 overlap: exchcd<->primaryexch
one-to-one (N1 A2 Q3 R4 B5 X0); shrcd reconstructed from sharetype (NS1/AD3/SB4/UG7) x
{issuertype REIT->8, FUND->4/5 by usincflg, EQTY->1/2 by usincflg}; 0.0000% unmapped in 2025.
Known seams (all measured, all outside or negligible for the SMT universe):
  * v2 uses a revised return methodology — on 2024 (both formats present) 6.2% of common-stock
    monthly returns differ by ~3bp median (max 0.39, prices identical). Applies to 2025 rows only.
  * v2 mthret embeds delisting returns natively; the legacy Shumway imputation (-35%/-55% when
    dlret missing) is not re-applied — it fired 2 times in 2020-2024 legacy data.
  * ADRs classified 32 (foreign-incorporated) in 2025 vs 31 in legacy; both outside shrcd 10/11/12.
  * dsf_v2 has no bidlo/askhi -> NaN for 2025 (unused by P1).

### Compustat annual/quarterly (m_aCompustat, a_aCompustat, m_QCompustat)
Overlap: <=0.6% (annual) / <=0.7% (quarterly) of any column's cells restated; 12-14 rows of
millions dropped; 14k-23k firm-months appended. Accepted under the revised-history policy.

### IBES
EPS estimates: ~0.1% of overlap cells changed (post-hoc analyst revisions).
Recommendations: 25,080 historical rows (~1.4k/yr recently) removed by the vendor — broker
contribution withdrawals; only 12% reappear under another analyst code. Affects the 4
recommendation factors' history; accepted (revised policy), flagged for the golden gate.
Actuals: 1,638 rows added, 47 dropped; negligible value changes.

### Fama-French / Liquidity
FF mkt/smb/hml revised on 0.2-2.6% of months by <=0.003 (routine republication); umd on ~50%
by <=0.016. monthlyLiquidity ps_innov revised on 100% of months (full-sample re-estimation —
exactly the look-ahead that motivated the PIT rebuild; the engine reads Data/ps_innov_pit.parquet
and does not consume this file).

### monthlyMarket (CRSP market index) — 2025 reconstructed from stock-level data
crsp.msi (published vwretd/ewretd/usdval) is frozen at 2024-12; no v2 index table exists on WRDS.
`DataDownloads/MarketReturnsV2Append.py` reconstructs 2025 from crsp.msf_v2: all securities,
value weights = prior month-end |prc|*shrout. Validated on 2022-01..2024-12 against the published
index: corr 0.99985 (vw & ew), max |diff| 0.0026 (vw) / 0.0044 (ew), usdval ratio 0.958. Legacy
rows unchanged. Only Beta (ewretd regressor) and market-volatility factors read this file; the
2025 values carry that documented ~0.3% return-level uncertainty.

### IBES link — pipeline ordering (fixed 2026-08-17)
IBESCRSPLink.py expands WRDS link ranges against monthlyCRSP.parquet's calendar, so it must run
AFTER the CRSP refresh; the first refresh pass ran it before, leaving the link (and thus
SignalMasterTable.tickerIBES and all 17 IBES factors) empty for 2025. Re-run after CRSPv2Append:
55,224 link rows and full tickerIBES coverage on 57,689 SMT rows in 2025. Correct order is now
documented here: CRSP (legacy + v2 append) -> IBESCRSPLink -> SignalMasterTable -> factor build.

### Refresh-exposed defects (2026-08-17, fixed; found by p1_refresh_factor_gate + 36-agent diagnosis)
* CRSPv2Append monthly `vol` scaling: msf_v2.mthvol is in shares (legacy msf.vol in hundreds) —
  2025 monthly volume was 100x too large (turnover 14 vs 0.13). Fixed to /1e6. VolumeTrend's
  full-sample trim had propagated the 2025 outliers back to 27,278 historical cells.
* CRSPdistributions frozen at 2024-12 (legacy crsp.msedist retired like msf/msi). NEW
  `DataDownloads/CRSPDistributionsV2Append.py` appends from crsp.stkdistributions with a distcd
  crosswalk measured on 2023-24 (45,099 joined rows): cd1/cd2/cd3 deterministic from
  distype/dispaymenttype/disfreqtype; cd4 (tax status, unread by P1) set to the legacy modal value.
* Cash: `_op_cash_rdq_signal` tie-break for two fiscal quarters sharing one rdq was keep='first'
  after sorting only on (gvkey, rdq) -> depended on m_QCompustat physical row order, which changed
  with the rebuild (282 historical cells flipped). Now sorts on datadateq (mergesort) and keeps the
  latest fiscal quarter. KERNEL_VERSIONS cash_rdq_signal=2.
* PriceDelay×3: reference forward-fills each July stamp until the permno's NEXT stamp — the fill
  length depends on future information; extending history by a year created 44,343 new valued
  cells in 2010-2024. Now a fixed 12-month PIT horizon (July..June). KERNEL_VERSIONS price_delay=2.
  Values on the reference's own support are unchanged; the deviation is support-only.
* CoskewACX: reference's anchored window reaches to a relisted stock's pre-gap rows; the inflated
  nobs then dominates the RELATIVE min-obs rule and dropped every other stock in 2025-07..12.
  Window now bounded to 12 calendar months. KERNEL_VERSIONS coskew_signal=2.
* IBESCRSPLink pipeline order and monthlyMarket 2025 reconstruction: see entries above.
Verified benign by independent adversarial re-derivation (60 factors): Compustat/IBES
restatements (incl. header-SIC reclassifications rewriting firm histories — a pre-existing
reference-recipe look-ahead now noted for GP/tang/BrandInvest/NetDebtPrice/PayoutYield/
NetPayoutYield/AbnormalAccruals: recommend sic_pit=coalesce(sich,sic)), Fama-French factor
republication rippling through every rolling-window/regression factor (BetaLiquidityPS,
ResidualMomentum, IdioVol3F, ReturnSkew3F, IdioVolAHT, BetaFP, Coskewness, PriceDelay values),
SignalMasterTable listing-span extension for 16 permnos relisted in 2025 (momentum/dividend
SPAN_OUT factors; reference-faithful), IBES adjusted-file rescaling after splits (EarningsStreak).

### PriceDelay reference forward-fill was an exploitable ex-post delisting flag (proven 2026-08-17)
Downstream consequence of the PriceDelay support look-ahead (fixed above): under the reference
rule, a stock's PriceDelay is NaN in Aug y..Jun y+1 exactly when it has NO later July stamp —
i.e. it is about to stop trading. A downstream ML pipeline that median-fills NaN and
cross-sectionally ranks maps that block to exactly 0.0, creating a clean "will delist within
<=11 months" flag. Verified on the Algo_Trading LightGBM backtest (20-agent drift analysis):
~78% of the L/S return gap between the Dec-2024 vintage (33.8%/yr) and the PIT-correct vintage
(23.0%/yr) on identical 2010-24 test months is P&L booked on exactly those leak-class stocks;
before PriceDelay entered the model (2010-13) the vintages are indistinguishable; ridge is
unchanged (prediction Spearman 0.998). The fixed 12-month PIT horizon removes the flag.
Lesson: support-only look-aheads are as dangerous as value look-aheads for nonlinear learners.
