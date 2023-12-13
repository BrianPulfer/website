import { useEffect } from 'react'
import { Box, Text, Link, Spinner } from '@chakra-ui/react'

import AppLayout from '@/components/Layout/AppLayout'

export default function GitHub (): JSX.Element {
  useEffect(() => {
    window.location.href = 'https://github.com/BrianPulfer'
  }, [])

  return (
    <Box
    height="100vh"
    display="flex"
    flexDirection="column"
    justifyContent="center"
    alignItems="center"
  >
    <Spinner size="xl"/>
    <Text mt={4} fontSize="lg" fontWeight="bold">
      Redirecting to <Link href={'https://github.com/BrianPulfer'} color={'blue.500'}>GitHub</Link>
    </Text>
  </Box>
  )
}

GitHub.getLayout = function getLayout (page: React.ReactElement) {
  return (
  <AppLayout>
    {page}
  </AppLayout>
  )
}
