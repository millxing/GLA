---
title: "Is clutch performance persistent?"
date: 2026-03-09
label: Team Analysis
summary: "Follow-up to: Lakers are getting it done when it counts"
tags: lakers, follow-up, clutch
visible: true
---

In a [2026-03-07 post on r/nbadiscussion](https://www.reddit.com/r/nbadiscussion/comments/1rnmz4l/lakers_are_getting_it_done_when_it_counts/) I observed that the 2025-26 Lakers win-loss record was a lot better than would be expected based on their mediocre net rating. As was pointed out in the comments, this is usually analyzed using the [pythagorean expectation](https://en.wikipedia.org/wiki/Pythagorean_expectation), an estimate for winning percentage based on scoring differential devised by Bill James. There will always be noise around this estimate, and it's typically assumed that residuals around the pythagorean expectation will mean revert. I also pointed out that the Lakers had very good performance in [clutch time](https://www.espn.com/nba/story/_/id/44816294/nba-clutch-player-year-winners-stats-more) and suggested that their win percentage might not be just lucky but possibly due to repeatable clutch performance. Given that they have Luka and LeBron, this didn't seem like a totally crazy idea.

Commenters noted that *of course* teams with positive pythagorean residuals have good clutch performance. How could they otherwise? I don't think I had fully appreciated that point until it was made. 

This and other comments raised a few questions:
- How much of an outlier was the Lakers pythagorean residual?
- How much of an outlier was the Lakers clutch performance?
- What is the historical relationship between pythagorean residuals and clutch performance?
- How persistent is clutch performance? Does good clutch performance in one period tell us anything about future clutch performance?

**I turned on STOCKTON (the AI integrated into my custom historical database) and had it crunch the numbers.**

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

![Year-to-year clutch persistence](/images/lakers_clutch_persistence_yoy.svg "Year-to-year clutch persistence")

Next, Stockton looked at persistence from the first half to the second half of the season. Does relative clutch performance in the first half of the season predict a team's relative clutch performance in the second half? There were 731 within-year team pairs (44 pairs were excluded for having fewer than 100 clutch possessions and at least 10 clutch games in each half). The correlation between relative clutch performance in the first half and relative clutch performance in the second half is 0.190. That's a lot higher than the year-to-year correlation, and it is very statistically significant (t-stat = 5.218, p < .0001). 

I think we've established that there is strong evidence for persistence between team-seasons and within team-seasons, with it being much stronger within season. That's encouraging for Lakers fans. [BTW: I was accused of being a Lakers fan-boy and that couldn't be more wrong. I'm a Celtics fan!]

![Within-season clutch persistence](/images/lakers_clutch_persistence_withinseason.svg "Within-season clutch persistence")

Here are some additional tables around the unusual nature of the Lakers season:

**Top 10 pythagorean residuals since 2000-01**

| Season | Team | Win pct | Pythagorean |
| --- | --- | --- | --- |
| 2022-23 | Milwaukee Bucks | 0.707 | 0.608 |
| 2020-21 | OKC Thunder | 0.306 | 0.206 |
| 2005-06 | Utah Jazz | 0.500 | 0.404 |
| 2015-16 | Golden State Warriors | 0.890 | 0.799 |
| 2007-08 | New Jersey Nets | 0.415 | 0.326 |
| 2015-16 | Memphis Grizzlies | 0.512 | 0.424 |
| 2021-22 | Portland Trail Blazers | 0.329 | 0.245 |
| 2025-26 | Los Angeles Lakers | 0.603 | 0.521 |
| 2017-18 | Cleveland Cavaliers | 0.610 | 0.532 |
| 2009-10 | Dallas Mavericks | 0.671 | 0.593 |

**Top 10 clutch-minus-overall net ratings since 2000-01**

| Season | Team | Clutch-minus-overall net rating |
| --- | --- | --- |
| 2020-21 | Portland Trail Blazers | 27.3 |
| 2025-26 | Los Angeles Lakers | 26.4 |
| 2021-22 | Phoenix Suns | 26.0 |
| 2023-24 | Chicago Bulls | 25.7 |
| 2015-16 | Golden State Warriors | 25.2 |
| 2010-11 | Dallas Mavericks | 25.2 |
| 2013-14 | Memphis Grizzlies | 25.0 |
| 2012-13 | Miami Heat | 25.0 |
| 2019-20 | Oklahoma City Thunder | 24.5 |
| 2008-09 | Cleveland Cavaliers | 23.6 |
