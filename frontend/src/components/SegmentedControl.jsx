/**
 * One toggle-group pattern for every view/scope switcher in the app.
 *
 * options: [{ value, label, disabled?, title? }]
 * Renders native buttons with aria-pressed so state is exposed to assistive tech,
 * and arrow keys move between options (roving focus within the group).
 */
function SegmentedControl({ options, value, onChange, label, size, block = false, className = '' }) {
  const enabled = options.filter((option) => !option.disabled)

  const handleKeyDown = (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const index = enabled.findIndex((option) => option.value === value)
    let next = index
    if (event.key === 'ArrowRight') next = (index + 1) % enabled.length
    if (event.key === 'ArrowLeft') next = (index - 1 + enabled.length) % enabled.length
    if (event.key === 'Home') next = 0
    if (event.key === 'End') next = enabled.length - 1
    const target = enabled[next]
    if (!target) return
    onChange(target.value)
    const group = event.currentTarget
    requestAnimationFrame(() => {
      group.querySelector(`[data-value="${CSS.escape(String(target.value))}"]`)?.focus()
    })
  }

  const classes = ['seg', size === 'sm' ? 'seg--sm' : '', block ? 'seg--block' : '', className]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={classes} role="group" aria-label={label} onKeyDown={handleKeyDown}>
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            className="seg-btn"
            data-value={option.value}
            aria-pressed={active}
            disabled={option.disabled}
            title={option.title}
            tabIndex={active || (!enabled.some((o) => o.value === value) && option === enabled[0]) ? 0 : -1}
            onClick={() => !active && onChange(option.value)}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

export default SegmentedControl
