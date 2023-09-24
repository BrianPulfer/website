import Navbar from './../Navbar'
import { Container } from '@chakra-ui/react'
import { Inter } from 'next/font/google'

export default function AppLayout ({ children }: { children: React.ReactNode }) {
  // const inter = Inter({ subsets: ['latin'] })

  return (
        <>
            <Navbar />
            <Container
                maxW={'container.lg'}
                className={'flex flex-col items-center justify-between m-0 '}
            >
                {children}
            </Container>
        </>
  )
}
