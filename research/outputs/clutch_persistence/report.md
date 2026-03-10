# Clutch Net Rating Persistence Study

## Setup

- Seasons analyzed: 2000-01 to 2025-26
- Team-seasons: 776
- Split-half samples: 776
- Year-to-year franchise pairs: 745
- Included playoffs: no
- Default within-season filter: >= 10 clutch games and >= 100 clutch possessions in each half
- Default year-to-year filter: >= 150 clutch possessions in each season

## Descriptive Sample Notes

- Median team-season clutch games: 41.000
- Median team-season clutch possessions: 326.000
- Median clutch time share: 3.749%11111
- Median raw clutch net-rating spread by season (std dev across teams): 11.261
- Median clutch-minus-non-clutch spread by season (std dev across teams): 9.985

## Headline Persistence Results

| Analysis | Metric | n | Pearson r | WLS slope | Control beta | Control non-clutch beta | Spearman-Brown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Within-season | Raw clutch net | 731 | 0.218 | 0.225 | 0.206 | 1.704 | 0.357 |
| Within-season | Clutch minus non-clutch | 731 | 0.190 | 0.198 | 0.194 | 1.755 | 0.319 |
| Year-to-year | Raw clutch net | 742 | 0.261 | 0.249 | 0.081 | 0.865 | n/a |
| Year-to-year | Clutch minus non-clutch | 742 | 0.103 | 0.095 | 0.078 | 0.371 | n/a |

## Permutation Null Benchmarks

| Analysis | Metric | Observed r | Null mean | Null 95% interval | Empirical p-value |
| --- | --- | ---: | ---: | --- | ---: |
| Within-season | Raw clutch net | 0.218 | -0.001 | [-0.076, 0.072] | 0.000 |
| Within-season | Clutch minus non-clutch | 0.190 | -0.002 | [-0.077, 0.071] | 0.000 |
| Year-to-year | Raw clutch net | 0.261 | 0.000 | [-0.073, 0.074] | 0.000 |
| Year-to-year | Clutch minus non-clutch | 0.103 | 0.000 | [-0.070, 0.073] | 0.006 |

## Data Integrity Checks

- Seasons loaded successfully: 26 / 26
- Max clutch-minus-all possession overage: 0.000
- Max clutch-minus-all minute overage: 0.000
- Max complement reconstruction error (possessions): 0.000
- Max complement reconstruction error (minutes): 0.000

## Sensitivity Summary

Headline correlations across the threshold grid are recorded in `summary_metrics.json` under `within_season.threshold_grid_results` and `year_to_year.threshold_grid_results` for thresholds 75, 100, 125, and 150 possessions.

## Plain-English Conclusion

Raw clutch performance shows some within-season persistence (r = 0.218) and some year-to-year carryover (r = 0.261).

After controlling for ordinary team strength with clutch-minus-non-clutch net rating, the persistence is still present within a season (r = 0.190) and still present from one season to the next (r = 0.103).

Relative to the no-persistence permutation null, the observed residual clutch signal is materially above random noise, so the better reading is that raw clutch results contain some real signal but a large share of apparent clutch dominance is explained by overall team quality and sampling noise.





Commenters noted that *of course* teams with positive pythagorean residuals have good clutch performance. How could they otherwise? I don't think I had fully appreciated that point until it was made. 

This and other comments raised a few questions:
- How much of an outlier was the Lakers pythagorean residual?
- How much of an outlier was the Lakers clutch performance?
- What is the historical relationship between pythagorean residuals and clutch performance?
- How persistent is clutch performance? Does good clutch performance in one period tell us anything about future clutch performance?


I turned on STOCKTON (the AI integrated into my custom historical database) and had it crunch the numbers.

These numbers are through March 7, 2026 and cover all seasons since 2000-01 (26 seasons):
- 2025-26 Lakers are 8th in pythagorean residual (.082, .603 vs expected .521).
- 2025-26 Lakers are also 8th in clutch net rating (+26.8).
- 2025-26 Lakers are 2nd in clutch minus overall net rating (+26.4).
- The top 10 pythagorean residuals averaged a 14.4 clutch-minus-overall net rating (10/10 were positive).
- The top 50 pythagorean residuals averaged a 9.3 clutch-minus-overall net rating (43/50 were positive).

So I think I was right to flag the Lakers season as unusual in both pythagorean residual and clutch performance (2nd best relative clutch performance ever). As was noted, there is a high degree of correlation between clutch performance and pythagorean residual. It's really hard to win games if you aren't very good and you don't step it up in clutch time.

What did Stockton find when it looked for evidence of clutch persistence? As many commenters noted, clutch time is a very small data sample, with only about 2% of the Lakers minutes coming in clutch situations. That makes it easy to mistake noise for a durable team trait.

It analyzed all seasons from 2000-01 to 2025-26 (regular seasons only), which produced 776 team-seasons.

First it looked at persistence from year to year. Does relative clutch performance, defined as clutch net rating minus overall net rating, in one year predict a team's relative clutch performance in the following year? There were 742 year-to-year team pairs (three pairs were excluded for having fewer than 150 clutch possessions). The correlation between relative clutch performance in year t and relative clutch performance in year t+1 is 0.103. That's not super high, but it is statistically significant (t-stat = 2.82, p < .0025). So there is evidence for some persistence year to year, despite the turnover that usually occurs between seasons. FWIW, the Lakers had a negative relative clutch rating last year.


Next, Stockton looked at persistence from the first half to the second half of the season. Does relative clutch performance in the first half of the season predict a team's relative clutch performance in the second half? There were 731 within-year team pairs (44 pairs were excluded for having fewer than 100 clutch possessions and at least 10 clutch games in each half). The correlation between relative clutch performance in the first half and relative clutch performance in the second half is 0.190. That's a lot higher than the year-to-year correlation, and it is very statistically significant (t-stat = 5.218, p < .0001). 

I think we've established that there is strong evidence for persistence between team-seasons and within team-seasons, with it being much stringer within season. That's encouraging for Lakers fans. [BTW: I was accused of being a Lakers fan-boy and that couldn't be more wrong. I'm a Celtics fan!]
