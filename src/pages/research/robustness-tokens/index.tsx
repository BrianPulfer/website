import AppLayout from '@/components/Layout/AppLayout'

import Head from 'next/head'
import { Image, Text, Link } from '@chakra-ui/react'
import BlogLayout from '@/components/Layout/BlogLayout'

export default function RobToks (): JSX.Element {
  return (
      <>
          <Head><title>Robustness Tokens</title></Head>
          <Text fontSize={'l'} textAlign={'right'}><b>Published:</b> 18.07.2024</Text>
          <Text fontSize={'5xl'} textAlign={'center'}>Robustness Tokens: Towards Adversarial Robustness of Transformers</Text>
          <Image mt={5} mb={5} src="/imgs/research/robustness-tokens/training.png" alt="Robustness Tokens"/>
          <Text> <Link textColor={'blue.500'} href='https://www.google.com'>Arxiv</Link></Text>
          <Text> <Link textColor={'blue.500'} href='https://www.google.com'>GitHub</Link></Text>
      </>
  )
}

RobToks.getLayout = function getLayout (page: React.ReactElement) {
  return (
      <AppLayout>
        <BlogLayout>
          {page}
        </BlogLayout>
      </AppLayout>
  )
}
