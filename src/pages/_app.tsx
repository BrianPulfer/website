import '@/styles/globals.css'

import Head from 'next/head'
import Script from 'next/script'
import type { AppProps } from 'next/app'

import theme from '../styles/theme'
import { ChakraProvider } from '@chakra-ui/react'

import '@fontsource/raleway/400.css'
import '@fontsource/open-sans/700.css'

export default function App ({ Component, pageProps }: { Component: any, pageProps: AppProps }): JSX.Element {
  const getLayout = Component.getLayout === undefined ? (page: React.ReactElement) => page : Component.getLayout
  const content = getLayout(<Component {...pageProps} />)

  // Initializing Google Analytics
  const GA_ID = 'G-2KSCVRRZHY'

  return (
    <>
        <Script
          strategy="lazyOnload"
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
        />

        <Script strategy="lazyOnload">
          {`
                      window.dataLayer = window.dataLayer || [];
                      function gtag(){dataLayer.push(arguments);}
                      gtag('js', new Date());
                      gtag('config', '${GA_ID}', {
                      page_path: window.location.pathname,
                      });
                  `}
        </Script>

        <Head>
          <meta name="viewport" content="initial-scale=1, width=device-width" />
        </Head>

      <ChakraProvider theme={theme}>
        {content}
      </ChakraProvider>
    </>
  )
}
