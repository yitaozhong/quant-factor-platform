# P1 Factor Engine Design

## Goal
Rebuild the `P1`-only stock-level factor library without relying on the Open Source Asset Pricing execution model.

Design target:
- handle the `171` P1-capable predictors
- minimize repeated scans and repeated joins
- keep peak memory bounded
- preserve point-in-time correctness
- support incremental reruns by family

## Core Principles
1. Separate source normalization from factor computation.
2. Compute shared rolling primitives once per family.
3. Keep storage columnar and partitioned by time.
4. Materialize only family-sized intermediates, not one giant all-factor table.
5. Standardize final outputs to a single long schema: `permno, yyyymm, factor, value`.

## Recommended Stack
- `DuckDB`: point-in-time joins, filtered scans, partition-aware SQL
- `Polars`: grouped lags, rolling windows, fast columnar transforms
- `NumPy/Numba`: custom rolling OLS kernels only where needed
- `Parquet + ZSTD`: all persistent intermediate and final tables

## Folder Layout
```text
Factor_Construction/
  factor_engine/
    config/
      universe.yaml
      storage.yaml
      factors_p1.yaml
    src/
      io/
        readers.py
        writers.py
        partitions.py
      pt/
        monthly_alignment.py
        quarterly_alignment.py
        ibes_alignment.py
      kernels/
        panel_lag.py
        rolling_monthly.py
        rolling_daily.py
        cross_section.py
        regressions.py
      families/
        monthly_returns.py
        daily_risk.py
        annual_accounting.py
        quarterly_accounting.py
        ibes_signals.py
        distributions.py
      registry/
        factor_registry.py
      pipeline/
        build_sources.py
        build_family.py
        build_all_p1.py
    data/
      raw/
      normalized/
        crsp_m/
        crsp_d/
        comp_a_pt/
        comp_q_pt/
        ibes_pt/
        crsp_dist/
      features/
        monthly_returns/
        daily_risk/
        annual_accounting/
        quarterly_accounting/
        ibes/
        distributions/
      output/
        by_factor/
        by_family/
    notebooks/
      validate_backbone.ipynb
      validate_factors.ipynb
```

## Canonical Source Tables
These are the only tables factor code should read directly.

### 1. `crsp_m`
Key: `(permno, month)`

Core columns:
- `permno, permco, month`
- `ret, retx`
- `prc, shrout`
- `vol`
- `bidlo, askhi`
- `mve_permno, mve_permco`
- `exchcd, shrcd`
- `sic_crsp`
- `dlret, dlstcd`

Use:
- size, price, momentum, reversal, volume, issuance, turnover, BM denominator support

### 2. `crsp_d`
Key: `(permno, date)`

Core columns:
- `permno, date`
- `ret`
- `prc, shrout, vol`
- `cfacpr, cfacshr`

Use:
- beta, idio vol, realized vol, maxret, illiquidity, trend, skew, delay, zero-trade

### 3. `comp_a_pt`
Key: `(permno, month)`

Definition:
- annual Compustat restated into monthly point-in-time availability rows

Core columns:
- identifiers: `permno, gvkey, datadate, month`
- accounting fields needed by annual factors
- precomputed `fyear`, `age_since_datadate`, `report_lag_months`

Use:
- accruals, profitability, investment, leverage, payout, valuation ratios

### 4. `comp_q_pt`
Key: `(permno, month)`

Definition:
- quarterly Compustat aligned to the first valid availability month

Core columns:
- identifiers: `permno, gvkey, datadate, rdq, month`
- quarterly fields needed by surprise, earnings, revenue, cash, `roaq`

Use:
- quarterly factor family

### 5. `ibes_pt`
Key: `(permno, month, horizon, measure_type)`

Definition:
- IBES summary history normalized to one monthly point-in-time panel

Core columns:
- `permno, ticker_ibes, month`
- `statpers, fpedats, fpi`
- `numest, meanest, medest, stdev`
- recommendation fields where applicable
- actuals support fields where applicable

Use:
- revisions, dispersion, FEPS, recommendations, analyst value, forecast-based signals

### 6. `crsp_dist`
Key: `(permno, exdt)`

Core columns:
- `permno, exdt, paydt, rcrddt`
- `divamt, distcd, facshr`

Use:
- dividend initiation, omission, seasonality, short-horizon yield

## Pipeline DAG
```text
raw P1 downloads
  -> normalize identifiers and dtypes
  -> build canonical source tables

crsp.msf/msenames/msedelist
  -> crsp_m
  -> monthly_returns family
  -> monthly market family support
  -> cross-sectional market features

crsp.dsf + ff daily
  -> crsp_d
  -> daily_risk shared kernels
  -> beta / skew / delay / volatility factors

comp.funda + CCM
  -> comp_a_pt
  -> annual_accounting shared kernels
  -> annual accounting factors

comp.fundq + CCM
  -> comp_q_pt
  -> quarterly_accounting shared kernels
  -> quarterly factors

IBES link + summaries + actuals + recs
  -> ibes_pt
  -> ibes shared kernels
  -> analyst / recommendation / FE factors

crsp.msedist
  -> crsp_dist
  -> distributions family

family outputs
  -> standardize schema
  -> write by_family parquet
  -> explode to by_factor parquet/csv only at the end
```

## Family Build Plan

### A. Monthly Returns Family
Input:
- `crsp_m`

Shared primitives:
- lagged returns: `l1, l2, ..., l60`
- rolling cumulative log returns
- rolling turnover and dollar volume
- rolling share changes
- month-level industry returns

Representative factors:
- `Size`, `Price`, `Mom6m`, `Mom12m`, `LRreversal`, `STreversal`
- `MomSeason`, `MomOffSeason`, `IndMom`, `IndRetBig`
- `ShareIss1Y`, `ShareIss5Y`, `CompEquIss`, `ShareVol`
- `DolVol`, `VolSD`, `VolMkt`, `std_turn`, `VolumeTrend`

Complexity:
- time: `O(N_monthly)`
- space: `O(active monthly partition + rolling buffers)`

### B. Daily Risk Family
Input:
- `crsp_d`
- daily Fama-French
- monthly market table where needed

Shared primitives:
- excess return series
- rolling month assignment
- grouped sufficient statistics for OLS
- rolling volatility, skewness, max return, Amihud ratio
- lagged market returns for price-delay regressions

Representative factors:
- `Beta`, `BetaFP`, `BetaLiquidityPS`, `BetaTailRisk`
- `IdioVol3F`, `IdioVolAHT`, `RealizedVol`
- `ReturnSkew`, `ReturnSkew3F`, `Coskewness`, `CoskewACX`
- `PriceDelaySlope`, `PriceDelayRsq`, `PriceDelayTstat`
- `High52`, `Illiquidity`, `MaxRet`, `TrendFactor`
- `zerotrade1M`, `zerotrade6M`, `zerotrade12M`

Complexity:
- time: near `O(N_daily)` plus grouped regressions with small `k`
- space: bounded by one or a few months of daily partitions

### C. Annual Accounting Family
Input:
- `comp_a_pt`
- `crsp_m` for market scaling

Shared primitives:
- 12/24/36/60 month lags
- annual deltas
- asset, sales, equity, debt, payout denominators
- trailing averages and rolling growth rates

Representative factors:
- `Accruals`, `TotalAccruals`, `PctAcc`, `PctTotAcc`, `dNoa`
- `AssetGrowth`, `Investment`, `InvestPPEInv`, `GrLTNOA`
- `BookLeverage`, `Leverage`, `CompositeDebtIssuance`, `NetDebtFinance`
- `GP`, `OperProf`, `CBOperProf`, `CF`, `cfp`, `EP`, `SP`, `BM`
- `ChEQ`, `ChInv`, `ChNWC`, `ChNNCOA`, `Tax`
- `RD`, `RDcap`, `AdExp`, `GrAdExp`, `BrandInvest`

Complexity:
- time: `O(N_annual_pt)`
- space: low; monthly PIT annual data is much smaller than daily data

### D. Quarterly Accounting Family
Input:
- `comp_q_pt`

Shared primitives:
- quarter-over-quarter and year-over-year deltas
- surprise versus lagged/seasonal benchmark
- quarter availability alignment

Representative factors:
- `Cash`, `ChTax`, `EarningsSurprise`, `EarnSupBig`
- `NumEarnIncrease`, `RevenueSurprise`, `roaq`

Complexity:
- time: `O(N_quarterly_pt)`
- space: low

### E. IBES Family
Input:
- `ibes_pt`
- `crsp_m`
- `comp_a_pt` where valuation formulas need accounting support

Shared primitives:
- forecast horizon splits by `fpi`
- within-firm revisions over defined windows
- dispersion and analyst-count changes
- recommendation changes
- forecast/actual alignment

Representative factors:
- `AnalystRevision`, `ForecastDispersion`, `ChNAnalyst`, `REV6`
- `ConsRecomm`, `ChangeInRecommendation`, `UpRecomm`, `DownRecomm`
- `FEPS`, `sfe`, `EarningsForecastDisparity`, `ExclExp`
- `AnalystValue`, `AOP`, `PredictedFE`, `EarningsStreak`, `fgr5yrLag`

Complexity:
- time: `O(N_ibes_pt)`
- space: medium; keep horizon-specific slices instead of loading all IBES forms together

### F. Distributions Family
Input:
- `crsp_dist`
- `crsp_m`

Shared primitives:
- classify distribution events by `distcd`
- align ex-dates to monthly factor dates
- track event history by permno

Representative factors:
- `DivInit`, `DivOmit`, `DivSeason`, `DivYieldST`

Complexity:
- time: `O(N_dist)`
- space: low

## Shared Kernel Layer
These kernels are where most of the engineering effort should go.

### Panel kernels
- `lag(panel, cols, n)`
- `delta(panel, cols, n)`
- `rolling_log_return(panel, window)`
- `rolling_sum/mean/std(panel, cols, window)`
- `growth_rate(panel, col, lag_n)`

### Cross-sectional kernels
- `month_rank(df, col)`
- `nyse_breakpoints(df, col)`
- `industry_aggregate(df, group_cols, value_col)`
- `winsorize_by_month(df, col)`

### Regression kernels
- `rolling_beta(daily_panel, factors, min_obs)`
- `rolling_residual_vol(daily_panel, factors, min_obs)`
- `price_delay_stats(daily_panel, n_lags, min_obs)`
- `cross_sectional_residual(month_panel, y, X)`

## Storage Strategy
Use three storage layers.

### 1. Normalized source layer
One table per source family:
- `normalized/crsp_m/year=YYYY/*.parquet`
- `normalized/crsp_d/year=YYYY/month=MM/*.parquet`
- `normalized/comp_a_pt/year=YYYY/*.parquet`
- `normalized/comp_q_pt/year=YYYY/*.parquet`
- `normalized/ibes_pt/year=YYYY/*.parquet`

### 2. Feature layer
Family-specific intermediate tables:
- `features/monthly_returns/`
- `features/daily_risk/`
- `features/annual_accounting/`
- `features/quarterly_accounting/`
- `features/ibes/`

These should contain reusable columns, not final factors only.

### 3. Output layer
Two formats:
- `output/by_family/*.parquet`
- `output/by_factor/factor_name.parquet`

Avoid writing CSV during compute. Write CSV only as final export if needed.

## Execution Order
1. Build `crsp_m`, `crsp_d`, `comp_a_pt`, `comp_q_pt`, `ibes_pt`, `crsp_dist`
2. Build `monthly_returns`
3. Build `annual_accounting`
4. Build `quarterly_accounting`
5. Build `ibes`
6. Build `distributions`
7. Build `daily_risk`

Reason:
- daily risk is the heaviest stage; run it after the lighter families are already validated
- monthly and accounting families expose most alignment bugs early

## Factor Registry
Each factor should be declared once in config, not hardcoded ad hoc.

Example registry entry:
```yaml
- factor: Mom12m
  family: monthly_returns
  inputs: [crsp_m]
  required_columns: [permno, month, ret]
  kernels: [rolling_log_return]
  params:
    start_lag: 2
    end_lag: 12
  output: output/by_factor/Mom12m.parquet
```

Example for a heavier factor:
```yaml
- factor: PriceDelayRsq
  family: daily_risk
  inputs: [crsp_d, ff_daily]
  required_columns: [permno, date, ret, mktrf, rf]
  kernels: [price_delay_stats]
  params:
    n_lags: 4
    min_obs: 26
  output: output/by_factor/PriceDelayRsq.parquet
```

## Space and Time Controls
To keep the build practical:

- partition daily data by `year/month`
- process one family at a time
- write intermediate features immediately after computation
- never keep more than one large daily family frame in memory
- prune columns aggressively before joins
- pre-sort once and reuse sorted partitions
- cache only canonical tables and high-value shared features

## Validation Strategy
For each family:
1. row-count sanity by month
2. permno coverage by month
3. null-rate by month
4. spot-check names like `NVDA` and `AMD`
5. compare wide-format benchmark files where available

Validation should be part of the pipeline, not a separate afterthought.

## First Implementation Slice
Build in this order:
1. canonical tables
2. monthly returns family
3. annual accounting family
4. daily beta/volatility subset

That gets a large share of the library with manageable implementation risk.
