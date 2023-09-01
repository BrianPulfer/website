import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { Inter } from 'next/font/google'
import Navbar from '@/components/Navbar'

import theme from "./theme"
import { ChakraProvider, Container } from '@chakra-ui/react'

import "@fontsource/raleway/400.css";
import "@fontsource/open-sans/700.css";

import "../styles/globals.css";

const inter = Inter({ subsets: ['latin'] })


export default function App({ Component, pageProps }: AppProps) {
  return (
    <ChakraProvider theme={theme}>
      <Navbar />
      <Container
        maxW={"container.lg"}
        className={`flex flex-col items-center justify-between m-0 ${inter.className}`}
      >
        <Component {...pageProps} />
      </Container>
    </ChakraProvider>
  )
}
