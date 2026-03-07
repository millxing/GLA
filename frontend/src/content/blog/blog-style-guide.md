---
title: "Blog Markdown Style Guide"
date: 2026-03-07
label: Documentation
summary: "Reference for writing blog posts in Markdown for the hidden Extra Pass Analytics blog."
tags: docs, markdown, blog
visible: false
---

# Blog post format

Use the Lakers post in [2026-03-07-lakers.md](/blog) as the model for structure and tone.

Start each post with front matter:

```md
---
title: "The Lakers Are Getting It Done When It Counts"
date: 2026-03-07
label: Team Analysis
summary: "A closer look at a strange Lakers season"
tags: lakers, clutch
visible: true
---
```

Recommended filename pattern:

```md
YYYY-MM-DD-short-slug.md
```

Example:

```md
2026-03-07-lakers.md
```

## Basic formatting

Bold:

```md
**the Lakers have posted a +26.8 net rating**
```

Italic:

```md
*timely offense*
```

Link:

```md
[NBA.com](https://www.nba.com)
```

Inline code:

```md
`frontend/public/images`
```

Blockquote:

```md
> The Lakers are consistently controlling endings.
```

## Headings and lists

Heading levels:

```md
# Main Heading
## Section Heading
### Subsection Heading
```

Bullet list:

```md
- Shot-making
- Free-throw rate
- Clutch execution
```

Numbered list:

```md
1. Set up the front matter
2. Write the summary
3. Add charts or video
```

## Images

Put blog images in:

```md
frontend/public/images/
```

Single image:

```md
![Lakers record vs net rating](/images/lal_2025-26_blog_record_vs_net.png)
```

Image with caption:

```md
![Lakers record vs net rating](/images/lal_2025-26_blog_record_vs_net.png "Record and net rating through March 7, 2026")
```

Side-by-side images on one line:

```md
![Fourth quarter chart](/images/lal_2025-26_blog_fourth_quarter.png "Fourth-quarter margin") ![Time splits chart](/images/lal_2025-26_blog_time_splits.png "Garbage time vs non-garbage time")
```

## Chart style

Blog charts should match the website visual language instead of using a separate "report" look.

Use these chart rules by default:

- Use a plain white background. Do not add decorative blobs, gradients, tinted backdrops, or other "funky" background treatments.
- Keep spacing tight. Avoid excess padding above the title, between subtitle and chart, and between the source line and the bottom edge of the image.
- Remove sidebar explainer boxes unless they are truly necessary. Prefer a cleaner chart-first layout.
- Keep typography simple and consistent with the site theme.
- Make the chart feel like part of the blog page, not like an exported slide deck.

Axis and annotation guidance:

- Rotate vertical y-axis labels 90 degrees when needed so they read cleanly.
- Prefer full labels over abbreviations when space allows. Example: use `Winning percentage` instead of `Win pct`.
- Put callouts in unused chart space rather than forcing them into the plot or into a separate right-hand box.

### Lakers chart notes

The Lakers post established these chart-specific conventions:

- `lal_2025-26_blog_fourth_quarter.png`
  - Keep the title.
  - Use the subtitle: `Average scoring margin by quarter`
  - Remove the text boxes on the right.
  - Use the source line: `Source: NBA linescore data.`

- `lal_2025-26_blog_record_vs_net.png`
  - Use the title: `Lakers' record better than expected given net rating`
  - Use the subtitle: `Through March 7, 2026.`
  - Remove the text boxes on the right.
  - Rotate the y-axis label 90 degrees.
  - Use `Winning percentage` as the y-axis label.
  - Move the Lakers callout with record and net rating into the upper-left quadrant where there is open space.

- `lal_2025-26_blog_time_splits.png`
  - Keep the clean site-matched theme.
  - Remove the text box on the right.
  - In the subtitle, use `mediocre` rather than `ordinary`.
  - Use the source line: `Source: NBA play-by-play data`

## Video embeds

YouTube embed:

```md
@[youtube](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
```

Local video embed from the public folder:

```md
@[video](/videos/lakers-breakdown.mp4)
```

## Suggested post structure

1. Open with the core observation.
2. Add one or two paragraphs of supporting context.
3. Place a chart where the argument needs evidence.
4. Use one or two subheads for the middle of the post.
5. Close with the playoff or big-picture implication.

## Lakers example outline

The current Lakers post works well because it follows a clear sequence:

1. The record and net rating do not match.
2. Non-garbage and clutch splits explain part of the gap.
3. Late-game shot-making and execution define the team.
4. The profile is both dangerous and unstable.
5. That tension is what makes the team interesting.
