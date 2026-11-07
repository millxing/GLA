import { useEffect } from 'react'
import { Link, useParams } from 'react-router-dom'
import { renderMarkdownBlocks, visibleBlogPosts } from '../utils/blogMarkdown.jsx'
import { BlogPostMeta } from './Blog'
import './Blog.css'

function BackLink() {
  return (
    <Link to="/blog" className="blog-back">
      <span aria-hidden="true">←</span> All posts
    </Link>
  )
}

function BlogPost() {
  const { slug } = useParams()
  const post = visibleBlogPosts.find((candidate) => candidate.slug === slug)

  useEffect(() => {
    document.title = `${post ? post.title : 'Post not found'} · Extra Pass Analytics`
  }, [post])

  if (!post) {
    return (
      <div className="blog-page">
        <BackLink />
        <div className="blog-empty card">
          <h1 className="page-title">Post not found</h1>
          <p>There is no post at this address.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="blog-page">
      <BackLink />
      <article className="blog-article card">
        <header className="blog-article-header">
          <h1 className="blog-article-title">{post.title}</h1>
          <BlogPostMeta post={post} />
          {post.summary ? <p className="blog-article-summary">{post.summary}</p> : null}
        </header>
        <div className="blog-post-body">{renderMarkdownBlocks(post.content, post.slug)}</div>
      </article>
    </div>
  )
}

export default BlogPost
