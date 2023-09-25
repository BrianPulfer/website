import Navbar from './../Navbar'
import { Container } from '@chakra-ui/react'

export default function AppLayout ({ children }: { children: React.ReactNode }): JSX.Element {
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
