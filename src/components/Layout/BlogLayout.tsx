import { Container } from '@chakra-ui/react'

export default function BlogLayout ({ children }: { children: React.ReactNode }): JSX.Element {
  return (
      <Container m={0} maxW={'100%'}>
          {children}
      </Container>
  )
}
