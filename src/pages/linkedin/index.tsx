import { useEffect } from 'react'
import { Box, Text, Link, Spinner } from '@chakra-ui/react'

import AppLayout from '@/components/Layout/AppLayout'

export default function LinkedIn (): JSX.Element {
  useEffect(() => {
    window.location.href = 'https://www.linkedin.com/in/brianpulfer/'
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
      Redirecting to <Link href={'https://www.linkedin.com/in/brianpulfer/'} color={'blue.500'}>LinkedIn</Link>
    </Text>
  </Box>
  )
}

LinkedIn.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
      {page}
    </AppLayout>
  )
}
