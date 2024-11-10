import { useEffect } from 'react'
import { Box, Text, Link, Spinner } from '@chakra-ui/react'

import AppLayout from '@/components/Layout/AppLayout'

export default function Scholar (): JSX.Element {
  useEffect(() => {
    window.location.href = 'https://scholar.google.com/citations?user=vfpT1UkAAAAJ'
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
      Redirecting to <Link href={'https://scholar.google.com/citations?user=vfpT1UkAAAAJ'} color={'blue.500'}>Scholar</Link>
    </Text>
  </Box>
  )
}

Scholar.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
      {page}
    </AppLayout>
  )
}
