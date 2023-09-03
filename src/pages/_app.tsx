import '@/styles/globals.css'
import type { AppProps } from 'next/app'

import theme from "./theme"
import { ChakraProvider } from '@chakra-ui/react'

import "@fontsource/raleway/400.css";
import "@fontsource/open-sans/700.css";

import "../styles/globals.css";


export default function App({ Component, pageProps }: { Component: any, pageProps: AppProps }) {
  const getLayout = Component.getLayout || ((page: React.ReactElement) => page)
  const content = getLayout(<Component {...pageProps} />)

  return (
    <ChakraProvider theme={theme}>
      {content}
    </ChakraProvider>
  )
}
