import { parseBlogPosts, renderMarkdownBlocks } from '../utils/blogMarkdown.jsx'
import './Blog.css'

const postModules = import.meta.glob('../content/blog/*.md', {
  eager: true,
  query: '?raw',
  import: 'default',
})

const posts = parseBlogPosts(postModules).filter((post) => post.visible)

function Blog() {
  return (
    <div className="blog-page">
      <div className="blog-shell">
        <section className="blog-hero">
          <h1 className="blog-title">Extra Pass Analytics Blog</h1>
          <p className="blog-subtitle">
            Observations about NBA team performance using advanced analytics.
          </p>
        </section>

        <section className="blog-feed">
          {posts.length > 0 ? (
            posts.map((post) => (
              <article className="blog-post card" key={post.slug}>
                <header className="blog-post-header">
                  <div>
                    <p className="blog-post-label">{post.label}</p>
                    <h2 className="blog-post-title">{post.title}</h2>
                  </div>
                </header>

                {post.summary ? <p className="blog-post-summary">{post.summary}</p> : null}
                {post.displayDate || post.tags.length > 0 ? (
                  <div className="blog-post-meta-row">
                    {post.displayDate ? <p className="blog-post-meta">{post.displayDate}</p> : null}
                    {post.tags.length > 0 ? (
                      <div className="blog-post-tags" aria-label="Post tags">
                        {post.tags.map((tag) => (
                          <span className="blog-post-tag" key={`${post.slug}-${tag}`}>
                            #{tag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <div className="blog-post-body">{renderMarkdownBlocks(post.content, post.slug)}</div>
              </article>
            ))
          ) : (
            <div className="blog-empty card">
              <p className="blog-post-label">No Posts Yet</p>
              <p className="blog-empty-copy">
                Add a Markdown file to <code>frontend/src/content/blog</code> to populate the hidden blog.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default Blog
