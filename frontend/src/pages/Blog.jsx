import { Link } from 'react-router-dom'
import { visibleBlogPosts } from '../utils/blogMarkdown.jsx'
import './Blog.css'

export function BlogPostMeta({ post }) {
  if (!post.displayDate && !post.label && post.tags.length === 0) return null

  return (
    <div className="blog-meta">
      {post.displayDate ? <time dateTime={post.date}>{post.displayDate}</time> : null}
      {post.label ? <span className="blog-meta-label">{post.label}</span> : null}
      {post.tags.length > 0 ? (
        <ul className="blog-tags" aria-label="Tags">
          {post.tags.map((tag) => (
            <li className="blog-tag" key={tag}>
              #{tag}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function Blog() {
  return (
    <div className="blog-page">
      <header className="page-header">
        <div className="page-title-row">
          <h1 className="page-title">Blog</h1>
        </div>
      </header>

      {visibleBlogPosts.length > 0 ? (
        <ul className="blog-index card">
          {visibleBlogPosts.map((post) => (
            <li className="blog-index-item" key={post.slug}>
              <h2 className="blog-index-title">
                <Link to={`/blog/${post.slug}`}>{post.title}</Link>
              </h2>
              <BlogPostMeta post={post} />
              {post.summary ? <p className="blog-index-summary">{post.summary}</p> : null}
            </li>
          ))}
        </ul>
      ) : (
        <div className="blog-empty card">
          <p>
            No posts yet. Add a Markdown file to <code>frontend/src/content/blog</code>.
          </p>
        </div>
      )}
    </div>
  )
}

export default Blog
