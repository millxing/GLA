function slugToTitle(slug) {
  return slug
    .split('-')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatDisplayDate(dateValue) {
  if (!dateValue) return ''

  const date = new Date(`${dateValue}T12:00:00`)
  if (Number.isNaN(date.getTime())) return dateValue

  return new Intl.DateTimeFormat('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  }).format(date)
}

function parseFrontMatter(raw) {
  const normalized = raw.replace(/\r\n/g, '\n').trim()

  if (!normalized.startsWith('---\n')) {
    return {
      meta: {},
      content: normalized,
    }
  }

  const closingIndex = normalized.indexOf('\n---\n', 4)
  if (closingIndex === -1) {
    return {
      meta: {},
      content: normalized,
    }
  }

  const frontMatter = normalized.slice(4, closingIndex)
  const content = normalized.slice(closingIndex + 5).trim()
  const meta = {}

  for (const line of frontMatter.split('\n')) {
    const separatorIndex = line.indexOf(':')
    if (separatorIndex === -1) continue

    const key = line.slice(0, separatorIndex).trim()
    const value = line.slice(separatorIndex + 1).trim()
    meta[key] = value.replace(/^"(.*)"$/, '$1')
  }

  return { meta, content }
}

function parseTags(tagsValue) {
  if (!tagsValue) return []

  return tagsValue
    .split(',')
    .map((tag) => tag.trim())
    .filter(Boolean)
    .map((tag) => tag.replace(/^#/, '').replace(/\s+/g, '-'))
}

function parseVisible(value) {
  if (typeof value !== 'string') return true
  return value.trim().toLowerCase() !== 'false'
}

function parseImageToken(token) {
  const match = token.match(/^!\[(.*?)\]\s*\((\S+?)(?:\s+"(.*?)")?\)$/)
  if (!match) return null

  const [, alt, src, caption] = match
  return {
    alt,
    src,
    caption: caption || '',
  }
}

function parseStandaloneImages(line) {
  const matches = line.match(/!\[.*?\]\(.*?\)/g)
  if (!matches?.length) return null

  const remainder = line.replace(/!\[.*?\]\(.*?\)/g, '').trim()
  if (remainder) return null

  const images = matches.map(parseImageToken).filter(Boolean)
  return images.length > 0 ? images : null
}

function parseTableCells(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function isTableSeparator(line) {
  if (!line?.includes('|')) return false

  const cells = parseTableCells(line)
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

function getYouTubeEmbedUrl(url) {
  try {
    const parsed = new URL(url)

    if (parsed.hostname === 'youtu.be') {
      const id = parsed.pathname.slice(1)
      return id ? `https://www.youtube.com/embed/${id}` : null
    }

    if (parsed.hostname.includes('youtube.com')) {
      if (parsed.pathname === '/watch') {
        const id = parsed.searchParams.get('v')
        return id ? `https://www.youtube.com/embed/${id}` : null
      }

      if (parsed.pathname.startsWith('/shorts/')) {
        const id = parsed.pathname.split('/')[2]
        return id ? `https://www.youtube.com/embed/${id}` : null
      }

      if (parsed.pathname.startsWith('/embed/')) {
        return url
      }
    }
  } catch {
    return null
  }

  return null
}

function parseEmbedDirective(line) {
  const youtubeMatch = line.match(/^@\[youtube\]\((.*?)\)$/)
  if (youtubeMatch) {
    const embedUrl = getYouTubeEmbedUrl(youtubeMatch[1])
    if (!embedUrl) return null

    return {
      type: 'youtube',
      src: embedUrl,
    }
  }

  const videoMatch = line.match(/^@\[video\]\((.*?)\)$/)
  if (videoMatch) {
    return {
      type: 'video',
      src: videoMatch[1],
    }
  }

  return null
}

function renderInlineMarkdown(text, keyPrefix) {
  const tokens = text
    .split(/(!\[.*?\]\s*\(.*?\)|\*\*.*?\*\*|__.*?__|\*.*?\*|_.*?_|`.*?`|\[.*?\]\s*\(.*?\))/g)
    .filter(Boolean)

  return tokens.map((token, index) => {
    const key = `${keyPrefix}-${index}`

    const imageMatch = token.match(/^!\[(.*?)\]\s*\((.*?)\)$/)
    if (imageMatch) {
      const [, alt, src] = imageMatch

      return <img key={key} src={src} alt={alt} className="blog-inline-image" loading="lazy" />
    }

    if (
      (token.startsWith('**') && token.endsWith('**')) ||
      (token.startsWith('__') && token.endsWith('__'))
    ) {
      return <strong key={key}>{token.slice(2, -2)}</strong>
    }

    if (
      (token.startsWith('*') && token.endsWith('*')) ||
      (token.startsWith('_') && token.endsWith('_'))
    ) {
      return <em key={key}>{token.slice(1, -1)}</em>
    }

    if (token.startsWith('`') && token.endsWith('`')) {
      return <code key={key}>{token.slice(1, -1)}</code>
    }

    const linkMatch = token.match(/^\[(.*?)\]\s*\((.*?)\)$/)
    if (linkMatch) {
      const [, label, href] = linkMatch
      const isExternal = /^https?:\/\//.test(href)

      return (
        <a
          key={key}
          href={href}
          target={isExternal ? '_blank' : undefined}
          rel={isExternal ? 'noreferrer' : undefined}
        >
          {renderInlineMarkdown(label, `${key}-label`)}
        </a>
      )
    }

    return token
  })
}

function renderImageFigure(image, key, className = '') {
  return (
    <figure className={className ? `blog-media-figure ${className}` : 'blog-media-figure'} key={key}>
      <img src={image.src} alt={image.alt} className="blog-inline-image" loading="lazy" />
      {image.caption ? <figcaption>{image.caption}</figcaption> : null}
    </figure>
  )
}

export function renderMarkdownBlocks(content, slug) {
  const lines = content.split('\n')
  const blocks = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index].trim()

    if (!line) {
      index += 1
      continue
    }

    const standaloneImages = parseStandaloneImages(line)
    if (standaloneImages) {
      if (standaloneImages.length === 1) {
        blocks.push(renderImageFigure(standaloneImages[0], `${slug}-image-${index}`))
      } else {
        blocks.push(
          <div className="blog-media-grid" key={`${slug}-gallery-${index}`}>
            {standaloneImages.map((image, imageIndex) =>
              renderImageFigure(image, `${slug}-gallery-${index}-${imageIndex}`, 'blog-media-figure--grid')
            )}
          </div>
        )
      }
      index += 1
      continue
    }

    const embed = parseEmbedDirective(line)
    if (embed) {
      if (embed.type === 'youtube') {
        blocks.push(
          <div className="blog-embed" key={`${slug}-youtube-${index}`}>
            <iframe
              src={embed.src}
              title="Embedded YouTube video"
              loading="lazy"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
        )
      } else if (embed.type === 'video') {
        blocks.push(
          <div className="blog-embed" key={`${slug}-video-${index}`}>
            <video controls preload="metadata">
              <source src={embed.src} />
              Your browser does not support the video tag.
            </video>
          </div>
        )
      }
      index += 1
      continue
    }

    if (line.includes('|') && isTableSeparator(lines[index + 1]?.trim())) {
      const headers = parseTableCells(line)
      const rows = []
      index += 2

      while (index < lines.length) {
        const rowLine = lines[index].trim()
        if (!rowLine || !rowLine.includes('|')) break
        rows.push(parseTableCells(rowLine))
        index += 1
      }

      blocks.push(
        <div className="blog-table-wrapper" key={`${slug}-table-${index}`}>
          <table className="blog-table">
            <thead>
              <tr>
                {headers.map((header, headerIndex) => (
                  <th key={`${slug}-table-head-${headerIndex}`}>
                    {renderInlineMarkdown(header, `${slug}-table-head-${headerIndex}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`${slug}-table-row-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`${slug}-table-cell-${rowIndex}-${cellIndex}`}>
                      {renderInlineMarkdown(cell, `${slug}-table-cell-${rowIndex}-${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
      continue
    }

    if (line.startsWith('### ')) {
      blocks.push(
        <h3 key={`${slug}-h3-${index}`}>{renderInlineMarkdown(line.slice(4), `${slug}-h3-${index}`)}</h3>
      )
      index += 1
      continue
    }

    if (line.startsWith('## ')) {
      blocks.push(
        <h2 key={`${slug}-h2-${index}`}>{renderInlineMarkdown(line.slice(3), `${slug}-h2-${index}`)}</h2>
      )
      index += 1
      continue
    }

    if (line.startsWith('# ')) {
      blocks.push(
        <h1 key={`${slug}-h1-${index}`}>{renderInlineMarkdown(line.slice(2), `${slug}-h1-${index}`)}</h1>
      )
      index += 1
      continue
    }

    if (line.startsWith('> ')) {
      const quoteLines = []
      while (index < lines.length && lines[index].trim().startsWith('> ')) {
        quoteLines.push(lines[index].trim().slice(2))
        index += 1
      }
      blocks.push(
        <blockquote key={`${slug}-quote-${index}`}>
          {renderInlineMarkdown(quoteLines.join(' '), `${slug}-quote-${index}`)}
        </blockquote>
      )
      continue
    }

    if (line.startsWith('- ') || line.startsWith('* ')) {
      const items = []
      while (index < lines.length) {
        const itemLine = lines[index].trim()
        if (!itemLine.startsWith('- ') && !itemLine.startsWith('* ')) break
        items.push(itemLine.slice(2))
        index += 1
      }

      blocks.push(
        <ul key={`${slug}-list-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={`${slug}-item-${itemIndex}`}>
              {renderInlineMarkdown(item, `${slug}-item-${itemIndex}`)}
            </li>
          ))}
        </ul>
      )
      continue
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = []
      while (index < lines.length) {
        const itemLine = lines[index].trim()
        if (!/^\d+\.\s+/.test(itemLine)) break
        items.push(itemLine.replace(/^\d+\.\s+/, ''))
        index += 1
      }

      blocks.push(
        <ol key={`${slug}-ordered-list-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={`${slug}-ordered-item-${itemIndex}`}>
              {renderInlineMarkdown(item, `${slug}-ordered-item-${itemIndex}`)}
            </li>
          ))}
        </ol>
      )
      continue
    }

    const paragraphLines = []
    while (index < lines.length) {
      const paragraphLine = lines[index].trim()
      if (
        !paragraphLine ||
        paragraphLine.startsWith('#') ||
        paragraphLine.startsWith('> ') ||
        paragraphLine.startsWith('- ') ||
        paragraphLine.startsWith('* ') ||
        /^\d+\.\s+/.test(paragraphLine) ||
        parseStandaloneImages(paragraphLine) ||
        parseEmbedDirective(paragraphLine)
      ) {
        break
      }
      paragraphLines.push(paragraphLine)
      index += 1
    }

    blocks.push(
      <p key={`${slug}-p-${index}`}>
        {renderInlineMarkdown(paragraphLines.join(' '), `${slug}-p-${index}`)}
      </p>
    )
  }

  return blocks
}

export function parseBlogPosts(postModules) {
  return Object.entries(postModules)
    .map(([path, raw]) => {
      const slug = path.split('/').pop().replace(/\.md$/, '')
      const { meta, content } = parseFrontMatter(raw)

      return {
        slug,
        title: meta.title || slugToTitle(slug),
        label: meta.label || 'Featured Post',
        summary: meta.summary || '',
        date: meta.date || '',
        displayDate: formatDisplayDate(meta.date),
        tags: parseTags(meta.tags),
        visible: parseVisible(meta.visible),
        content,
      }
    })
    .sort((a, b) => (a.date < b.date ? 1 : -1))
}
