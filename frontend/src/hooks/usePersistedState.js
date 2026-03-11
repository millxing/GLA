import { useState, useEffect } from 'react'
import { ENABLE_MODULE_STATE_PERSISTENCE } from '../config/featureFlags'

const STORAGE_PREFIX = 'gla_'

function resolveDefaultValue(defaultValue) {
  return typeof defaultValue === 'function' ? defaultValue() : defaultValue
}

/**
 * A useState hook that persists the value to localStorage.
 * Values are stored independently per page using unique keys.
 *
 * @param {string} key - The storage key (will be prefixed with 'gla_')
 * @param {any} defaultValue - The default value if nothing is stored
 * @returns {[any, Function]} - Same as useState: [value, setValue]
 */
export function usePersistedState(key, defaultValue) {
  const storageKey = STORAGE_PREFIX + key
  const [value, setValue] = useState(() => {
    if (!ENABLE_MODULE_STATE_PERSISTENCE) {
      return resolveDefaultValue(defaultValue)
    }

    try {
      const stored = localStorage.getItem(storageKey)
      return stored !== null ? JSON.parse(stored) : resolveDefaultValue(defaultValue)
    } catch {
      return resolveDefaultValue(defaultValue)
    }
  })

  useEffect(() => {
    try {
      if (!ENABLE_MODULE_STATE_PERSISTENCE) {
        localStorage.removeItem(storageKey)
        return
      }

      localStorage.setItem(storageKey, JSON.stringify(value))
    } catch {
      // Silently fail if localStorage is unavailable
    }
  }, [storageKey, value])

  return [value, setValue]
}
