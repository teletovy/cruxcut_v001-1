// components/theme-provider.tsx
'use client'

import * as React from 'react'
import { ThemeProvider as NextThemesProvider, type ThemeProviderProps } from 'next-themes'

export function ThemeProvider(props: ThemeProviderProps) {
  return <NextThemesProvider {...props} />
}