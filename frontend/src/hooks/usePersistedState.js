import { useState, useEffect } from 'react'

const STORAGE_PREFIX = 'gla_'

/**
 * A useState hook that persists the value to localStorage.
 * Values are stored independently per page using unique keys.
 *
 * @param {string} key - The storage key (will be prefixed with 'gla_')
 * @param {any} defaultValue - The default value if nothing is stored
 * @returns {[any, Function]} - Same as useState: [value, setValue]
 */
export function usePersistedState(key, defaultValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_PREFIX + key)
      return stored !== null ? JSON.parse(stored) : defaultValue
    } catch {
      return defaultValue
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value))
    } catch {
      // Silently fail if localStorage is unavailable
    }
  }, [key, value])

  return [value, setValue]
}
