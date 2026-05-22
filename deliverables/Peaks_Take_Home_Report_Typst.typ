#let navy = rgb("#17212b")
#let ink = rgb("#24211d")
#let muted = rgb("#6f6a60")
#let gold = rgb("#b98b2d")
#let paper = rgb("#fbfaf6")
#let wash = rgb("#f1eee7")
#let hairline = rgb("#d9d1c2")

#set page(paper: "a4", margin: (x: 1.55cm, y: 1.35cm))
#set text(font: ("New Computer Modern", "Libertinus Serif"), size: 9.35pt, lang: "en", fill: ink)
#set par(justify: true, leading: 0.62em, spacing: 0.72em)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #v(0.75em)
  #align(center)[#text(size: 14.2pt, weight: "bold", fill: navy)[#it.body]]
  #v(0.05em)
  #align(center)[#line(length: 62%, stroke: 0.55pt + gold)]
  #v(0.18em)
]
#show heading.where(level: 2): it => [
  #v(0.45em)
  #align(center)[#text(size: 10.6pt, weight: "bold", fill: navy)[#it.body]]
  #v(0.05em)
]
#show table: set text(size: 7.25pt)
#show table.cell.where(y: 0): set text(weight: "bold", fill: white)
#show table.cell.where(y: 0): set table.cell(fill: navy)

#let metric(label, value, note) = block(
  width: 100%,
  fill: paper,
  stroke: 0.45pt + hairline,
  radius: 3pt,
  inset: (x: 7pt, y: 6pt),
)[
  #align(center)[
    #text(size: 16.5pt, weight: "bold", fill: navy)[#value]
    #linebreak()
    #text(size: 6.6pt, fill: muted, tracking: 0.35pt)[#label]
    #linebreak()
    #text(size: 7.1pt, fill: muted)[#note]
  ]
]


#block(
  width: 100%,
  fill: wash,
  stroke: 0.55pt + hairline,
  radius: 4pt,
  inset: (x: 14pt, y: 12pt),
)[
  #align(center)[
    #text(size: 6.8pt, fill: muted, tracking: 1.1pt)[SENIOR DATA SCIENTIST TAKE-HOME]
    #linebreak()
    #v(0.18em)
    #text(size: 24pt, weight: "bold", fill: navy)[Peaks Marketing Performance]
    #linebreak()
    #text(size: 16pt, fill: navy)[Attribution, early LTV, and next-quarter budget choices]
    #v(0.35em)
    #line(length: 34%, stroke: 1pt + gold)
  ]
]

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  metric("TOTAL SPEND", "$3.99M", "730 days"),
  metric("INSTALLS", "408,010", "all platforms"),
  metric("IOS PAID SHARE", "6.3%", "MMP last-click"),
  metric("SKAN / MMP IOS", "6.7x", "paid install gap"),
)

= Executive Summary

Peaks spent \$3,993,866.19 across the period and generated 408,010 installs, 111,693 registrations, and 86,529 transacting users. 78,511 users show funding activity within the first 14 days. CMO read: last-click is fine for reconciliation, but budget choices should move to cohort value, iOS privacy correction, and incrementality tests.

The attribution gap should be thought of as a reliability check on whether the source labels can support budget decisions. iOS looks under-attributed because last-click labels only 6.3% of iOS installs as paid, versus 31.8% on Android, while Organic is 72.9% of iOS installs versus 47.4% on Android. SKAN reports 78,000 paid iOS installs, about 6.7x the MMP-paid iOS installs. Use SKAN as the paid iOS volume anchor, IPW LTV as the base value estimate, and the conversion-value mapping only as sensitivity until the real SKAN schema is available.

On value, the strongest 12-month fee-revenue cohorts are Google Non-Brand Search, Referral, Google Brand Search. Google Brand Search and Referral look efficient in last-click CAC, but Brand Search is likely harvesting demand created elsewhere. Protect Referral if marginal quality holds and protect Non-Brand Search under payback caps. Keep Meta on LTV bid caps, tighten App Campaigns unless campaign-level splits improve, and reduce TikTok pending a funded-AUM holdout.

The LightGBM early-LTV model uses only first-14-day data and its top predicted decile captures 99.7% of validation first-year fee revenue. That gives enough separation for bid caps and cohort steering, but the model should not be treated as proof of incremental lift.

#block(fill: rgb("#f4f7f9"), stroke: 0.7pt + rgb("#577590"), inset: 8pt, radius: 2pt)[Immediate moves: launch a weekly cohort value scorecard, replace iOS MMP-only reporting with SKAN-volume plus IPW-value reporting, and put spend guardrails on Brand Search, App Campaigns, and TikTok before the next budget increase.]

#pagebreak()

= Data And Method

The analysis starts from the raw CSVs and parses dates, timestamps, booleans, spend, event counts, transaction amounts, and SKAN postbacks with explicit Polars schemas. Registered and transacting users all join back to app events, so the main caveat is attribution quality rather than missing customer records.

#align(center)[#table(
  columns: (auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Table]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Rows / coverage]],
  [Marketing Spend],
  [6,569],
  [App Events],
  [962,811],
  [User Profiles],
  [111,693],
  [User Transactions],
  [581,906],
  [Skan Attribution Ios],
  [21,766],
  [Installed users],
  [408,010],
  [Registered users],
  [111,693],
  [Transactions in profiles],
  [100.0%],
)]

I define LTV as first-year fee revenue from AUM: running balance times the current 0.5% annual management fee, prorated over observed balance intervals. Users without a full 365-day horizon are excluded from supervised training.

= Part 1: Attribution And Channel Performance

The last-click view rewards channels close to the decision. Google Brand Search and Referral have the lowest CACs, while TikTok has the highest CPI and CAC. TikTok also has the weakest early funded rate, while Referral and Non-Brand Search convert registered users into funded users much more often. That ranking is incomplete because it mixes intent capture, referrals, prospecting, and brand activity into one last-click scorecard.

#figure(
  image("figures/cost_metrics.png", width: 86%),
  caption: [Last-click CAC rewards intent capture and referral mechanics; it is not enough for budget allocation.],
)

#align(center)[#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Channel]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Spend]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Installs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Regs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[CPI]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[CAC]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Install -> Reg]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Early Funded]],
  [Google Brand Search],
  [\$98,355.57],
  [12,238],
  [3,032],
  [\$8.04],
  [\$32.44],
  [24.8%],
  [69.9%],
  [Referral],
  [\$1,428,150.14],
  [84,634],
  [38,079],
  [\$16.87],
  [\$37.50],
  [45.0%],
  [79.0%],
  [Meta],
  [\$860,512.40],
  [28,207],
  [4,260],
  [\$30.51],
  [\$202.00],
  [15.1%],
  [63.2%],
  [Google Non-Brand Search],
  [\$385,620.13],
  [12,527],
  [1,891],
  [\$30.78],
  [\$203.92],
  [15.1%],
  [74.8%],
  [Google App Campaigns],
  [\$584,679.29],
  [15,834],
  [2,318],
  [\$36.93],
  [\$252.23],
  [14.6%],
  [62.5%],
  [TikTok],
  [\$636,548.66],
  [14,186],
  [2,150],
  [\$44.87],
  [\$296.07],
  [15.2%],
  [44.2%],
  [Organic],
  [\$0.00],
  [240,384],
  [59,963],
  [n/a],
  [n/a],
  [24.9%],
  [66.4%],
)]

This is not enough information to prove causal attribution. Rather, it is a check on whether the attribution labels are reliable enough for budget decisions. iOS looks under-attributed because its paid share is much lower than Android (6.3% versus 31.8%), Organic is much higher (72.9% versus 47.4%), and SKAN reports substantially more paid iOS installs than the MMP source field.

#figure(
  image("figures/attribution_mix.png", width: 86%),
  caption: [iOS has a much higher Organic share and a much lower paid-attributed share than Android.],
)

Paid spend and later Organic installs are strongly correlated, peaking at a 0-day lag (0.90). The shape matters: it is not a smooth decay curve. It drops after the first few days, rises again around a weekly lag, and then repeats. That points to shared campaign pacing or day-of-week structure, not a clean estimate of causal lift. Practical read: Organic is not behaving like an independent unpaid baseline.

#figure(
  image("figures/organic_lag.png", width: 86%),
  caption: [Organic installs move with paid spend on a weekly rhythm, not as a clean causal decay curve.],
)

== Attribution Anomaly Signals

The anomaly checks do not prove incrementality. They show that last-click labels vary across platform, tracking state, SKAN vs MMP, and time. That leaves last-click fine for bookkeeping, but risky for budget allocation.

#align(center)[#table(
  columns: (auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Signal]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Evidence]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Readout]],
  [Paid/Organic daily co-movement],
  [marketing\_spend + app\_events: best lag r=0.90 at 0 days],
  [Organic is not a clean unpaid baseline.],
  [iOS source mix discontinuity],
  [app\_events: paid 6.3% iOS vs 31.8% Android; Organic 72.9% iOS vs 47.4% Android],
  [Platform-level last-click labels are not comparable.],
  [SKAN exceeds MMP paid iOS],
  [skan + app\_events: 78,000 SKAN paid iOS installs vs 11,626 MMP],
  [Use SKAN to calibrate iOS paid volume.],
  [Non-tracking collapses to Organic],
  [user\_profiles + app\_events: non-tracking Organic share is 100.0% on iOS and 100.0% on Android],
  [Consent state affects attribution availability.],
  [Organic has real funded users],
  [app\_events + transactions: 59,963 Organic regs; 66.4% early funded],
  [Zero assigned spend does not mean zero acquisition influence.],
  [Referral is not auction media],
  [marketing\_spend: \$1,428,150 Referral spend with 0 impressions and 0 clicks],
  [Separate incentive/referral economics from paid-media CPI.],
)]

= Part 2: Early-LTV Prediction

The native LightGBM model uses only fields available within 14 days after install: install source, campaign, platform, country, registration/profile fields if the user registered, and first-14-day transaction behavior. Training uses older installs with a full 365-day observation window, then validates on later eligible cohorts.

The two-stage LightGBM model improves over a transparent segment baseline. Validation MAE is \$6.91 versus \$7.45; log RMSE is 0.22 versus 0.25. The classifier AUC for positive LTV is 1.00.

The feature ranking fits the business model. Early funding behavior carries most of the signal: net deposit, gross deposit, transaction count, deposit count, balance, and timing to first transaction. That makes sense for Peaks because revenue comes from AUM fees; early balance formation is the best early clue about future fee revenue. One caveat: gender appears predictive in this dataset, but I would treat it as diagnostic, not automatically operational. A real investment-product bidding system should review demographic features with compliance and fairness constraints before use.

#figure(
  image("figures/decile_lift.png", width: 86%),
  caption: [After users are ranked by predicted LTV, revenue is concentrated in the highest-ranked users.],
)

== Most Predictive Feature Groups

#align(center)[#table(
  columns: (auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Feature]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[MAE Increase]],
  [net\_deposit\_14d],
  [\$11.82],
  [txn\_count\_14d],
  [\$7.41],
  [gross\_deposit\_14d],
  [\$6.64],
  [gender],
  [\$3.34],
  [deposit\_count\_14d],
  [\$1.85],
  [max\_balance\_14d],
  [\$0.47],
  [days\_to\_first\_transaction],
  [\$0.14],
  [days\_to\_registration],
  [\$0.09],
  [tracking\_enabled\_num],
  [\$0.08],
  [source],
  [\$0.03],
)]

Re-scoring channels by predicted value changes the CPI story. Google Non-Brand Search has the highest predicted LTV per registration, while Referral has the strongest predicted LTV per install and the fastest observed payback on complete cohorts. Google Brand Search still looks cheap by CPI and CAC, but that is probably demand capture rather than demand creation. TikTok is the weakest read here: high acquisition cost and low predicted customer value.

== Predicted Channel Quality

#align(center)[#table(
  columns: (auto, auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Channel]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Installs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Regs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Pred LTV / Install]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Pred LTV / Reg]],
  [Google Non-Brand Search],
  [12,527],
  [1,891],
  [\$10.84],
  [\$71.82],
  [Referral],
  [84,634],
  [38,079],
  [\$30.98],
  [\$68.85],
  [Google Brand Search],
  [12,238],
  [3,032],
  [\$14.47],
  [\$58.42],
  [Organic],
  [240,384],
  [59,963],
  [\$10.01],
  [\$40.13],
  [Meta],
  [28,207],
  [4,260],
  [\$5.60],
  [\$37.08],
  [Google App Campaigns],
  [15,834],
  [2,318],
  [\$5.24],
  [\$35.80],
  [TikTok],
  [14,186],
  [2,150],
  [\$2.94],
  [\$19.39],
)]

#figure(
  image("figures/channel_value.png", width: 86%),
  caption: [Predicted customer value and acquisition cost point to a different story than CPI alone.],
)

= iOS And SKAN Estimation

SKAN sees far more paid iOS volume than the MMP last-click source field. I treat that as a missing-data and selection-bias problem, not as proof of incremental lift. SKAN gives aggregate paid iOS volume, while user-level MMP data only sees a biased subset of paid iOS users. A cross-fitted LightGBM propensity model estimates each complete-cohort iOS user's chance of being observable as MMP-paid, then inverse-probability weights the observed paid iOS users before applying value to SKAN volume.

I use the IPW estimate as the primary planning number because it corrects the observable selection problem directly. The conversion-value rank calibration stays in the table as a sensitivity check because the true SKAN conversion schema is not available in the assignment data.

The comparison changes the read: IPW puts these networks at roughly \$3.96 to \$23.13 per SKAN install, while the rank-based conversion-value method is 1.3x to 7.3x higher. I would treat the CV method as upside sensitivity, not the base case.

#align(center)[#table(
  columns: (auto, auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Network]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[SKAN Installs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[MMP iOS Paid]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Gap]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[SKAN / MMP]],
  [Meta],
  [26,442],
  [3,926],
  [22,516],
  [6.74],
  [Google App Campaigns],
  [14,920],
  [2,151],
  [12,769],
  [6.94],
  [TikTok],
  [13,415],
  [2,003],
  [11,412],
  [6.70],
  [Google Non-Brand Search],
  [11,734],
  [1,784],
  [9,950],
  [6.58],
  [Google Brand Search],
  [11,489],
  [1,762],
  [9,727],
  [6.52],
)]

#align(center)[#table(
  columns: (auto, auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Network]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[SKAN Installs]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[IPW LTV / SKAN Install]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[CV LTV / SKAN Install]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[CV / IPW]],
  [Google Brand Search],
  [11,489],
  [\$23.13],
  [\$64.50],
  [2.79],
  [Google Non-Brand Search],
  [11,734],
  [\$21.80],
  [\$77.49],
  [3.56],
  [Meta],
  [26,442],
  [\$9.62],
  [\$33.55],
  [3.49],
  [TikTok],
  [13,415],
  [\$7.20],
  [\$9.56],
  [1.33],
  [Google App Campaigns],
  [14,920],
  [\$3.96],
  [\$28.99],
  [7.31],
)]

= Part 3: Recommendations

Budget allocation should move from CPI to expected fee revenue and payback, with separate rules for channel roles. Recommended move: protect Referral and Non-Brand Search, cap Brand Search credit, hold Meta under LTV guardrails, tighten App Campaigns, and reduce TikTok until it proves incremental funded AUM.

#align(center)[#table(
  columns: (auto, auto, auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Channel]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Action]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Value read]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Pred LTV / Install]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Payback Years]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[IPW iOS LTV]],
  [Google App Campaigns],
  [Tighten or reduce],
  [Overvalued if treated as scaled performance],
  [\$5.24],
  [6.41],
  [\$3.96],
  [Google Brand Search],
  [Cap and de-credit],
  [Overvalued by last-click],
  [\$14.47],
  [0.53],
  [\$23.13],
  [Google Non-Brand Search],
  [Protect / test more],
  [Undervalued by CPI-only views],
  [\$10.84],
  [2.33],
  [\$21.80],
  [Meta],
  [Hold / optimize],
  [Possibly undervalued on iOS, weak on raw payback],
  [\$5.60],
  [4.85],
  [\$9.62],
  [Organic],
  [Measure, do not treat as free],
  [Mixed bucket, partly privacy-masked demand],
  [\$10.01],
  [n/a],
  [n/a],
  [Referral],
  [Increase if scalable],
  [Undervalued by paid-media scorecards],
  [\$30.98],
  [0.47],
  [n/a],
  [TikTok],
  [Reduce pending test],
  [Overvalued by cheap-reach logic],
  [\$2.94],
  [13.52],
  [\$7.20],
)]

I classify channel roles from campaign names, media mechanics, and cohort value. Low CPM with awareness naming points to reach, high-CTR Search points to intent capture, Referral spend behaves like an incentive cost, and Organic is a mixed attribution bucket. That is why the KPI changes by channel: not every channel should be judged on CPI or last-click CAC alone.

#align(center)[#table(
  columns: (auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Channel]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Role]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Measurement note]],
  [Google App Campaigns],
  [Algorithmic app acquisition],
  [Performance channel, but iOS privacy loss means MMP counts are incomplete.],
  [Google Brand Search],
  [Brand demand harvesting],
  [Very efficient last-click economics, but likely captures demand created elsewhere.],
  [Google Non-Brand Search],
  [High-intent category demand capture],
  [Higher CPC/CPM, but strong LTV can justify spend if incrementality holds.],
  [Meta],
  [Scaled paid prospecting],
  [Broad reach with better CTR than TikTok; should be judged on cohort value and incrementality.],
  [Organic],
  [Unattributed, owned, and privacy-masked demand],
  [Do not assume it is unpaid; split owned baseline from paid spillover using tests.],
  [Referral],
  [Member-get-member acquisition],
  [High conversion and quality; treat separately from paid media because spend is an incentive cost.],
  [TikTok],
  [Upper-funnel reach / brand demand creation],
  [Low CPM and campaign naming point to awareness; last-click will miss assisted demand.],
)]

== Forward Measurement Framework

Operating rule: keep last-click for reconciliation, but do not let it choose the budget. Budget decisions should come from cohort value, iOS privacy correction, and incrementality tests. This avoids treating Organic as free, Brand Search as pure growth, or SKAN as user-level truth.

#align(center)[#table(
  columns: (auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Layer]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Primary metrics]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Cadence]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Decision use]],
  [Operational reconciliation],
  [Spend, impressions, clicks, installs, registrations, CPI, CAC],
  [Daily / weekly],
  [Finance, pacing, QA; not budget allocation by itself.],
  [Cohort economics],
  [Funded rate, predicted 365-day LTV, observed fee revenue, payback, ROAS],
  [Weekly cohorts],
  [Bid caps, budget guardrails, and channel quality ranking.],
  [iOS privacy correction],
  [SKAN installs, SKAN / MMP gap, IPW LTV per SKAN install, CV sensitivity],
  [Weekly / monthly],
  [Use SKAN as iOS volume anchor and IPW as the value base case.],
  [Incrementality testing],
  [Lift in registrations, funded users, net deposits, AUM, fee revenue, payback],
  [Monthly / quarterly],
  [Validate Brand Search credit, TikTok scale, and major budget shifts.],
  [Channel-role scorecards],
  [Search payback, referral marginal quality, prospecting LTV caps, brand lift],
  [Weekly review, monthly reset],
  [Stop comparing brand, referral, search, and prospecting on CPI alone.],
  [Organic baseline],
  [Organic trend, paid/Organic lag correlation, platform mix, tracking mix],
  [Monthly],
  [Prevent privacy-masked paid demand from being treated as free growth.],
)]

== Quick Wins

These are the first moves I would make before a broader attribution rebuild. They use the current data products and directly address the weak spots found in the analysis.

#align(center)[#table(
  columns: (auto, auto, auto, auto),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Quick win]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Owner]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[First action]],
  table.cell(fill: navy)[#text(fill: white, weight: "bold")[Success metric]],
  [Launch a weekly cohort value scorecard],
  [Growth Analytics + Finance],
  [Publish channel x platform x campaign cohorts with CPI, CAC, funded rate, predicted LTV, observed LTV, and payback.],
  [Weekly budget review uses predicted LTV and payback before spend moves.],
  [Replace iOS MMP-only reporting],
  [Marketing Analytics + Mobile Measurement],
  [Use SKAN as paid iOS volume anchor, IPW LTV as the base value estimate, and CV-rank only as sensitivity.],
  [Every iOS channel readout includes SKAN / MMP gap, IPW LTV, and CV sensitivity.],
  [Put spend guardrails on weak or ambiguous channels],
  [Growth Lead],
  [Cap Brand Search credit, tighten App Campaigns, and pause TikTok scale-ups until a holdout proves funded-AUM lift.],
  [No increase clears unless it passes LTV bid caps or an incrementality test.],
)]

The team does not need perfect attribution to make better budget calls. Start by changing the reporting grain, correcting iOS volume and value, and putting guardrails on channels where last-click is most likely to mislead.

Brand Search should have a smaller credit window or be reported as demand capture unless a holdout shows lift. Referral should stay outside paid-media auction reporting because its economics are incentive-led. For iOS, SKAN should anchor volume, IPW should be the base value estimate, and the conversion-value mapping should stay as sensitivity until the real schema is available.

= Limitations And Next Steps

The LightGBM model is fit for ranking cohorts and setting bid guardrails, but it predicts value from historical behavior; it does not prove incremental lift. Training also excludes newer installs without a full 365-day observation window, so calibration should be monitored as channel mix and campaign strategy change. Gender appears predictive in this dataset, but I would keep it diagnostic unless compliance and fairness review approves operational use.

LTV is based on observed AUM balances and the current 0.5% fee, so fee changes, balance mix changes, or product changes would need a refresh. The SKAN IPW correction assumes observed features explain MMP-paid observability; it does not recover unobserved network assignment or causal lift. The conversion-value mapping is a sensitivity check and should be replaced with the actual SKAN schema if the marketing team can provide it.

The budget recommendations are guardrails, not a guarantee that marginal spend will perform the same way. Referral and Non-Brand Search may saturate, Brand Search may keep harvesting demand created elsewhere, and Organic remains a mixed bucket of owned demand and privacy-masked paid demand. Next: pair the new reporting system with holdouts and cohort monitoring before making large permanent reallocations.
