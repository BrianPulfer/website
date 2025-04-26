import AppLayout from '@/components/Layout/AppLayout'
import BlogLayout from '../layout'
import Head from 'next/head'
import { Center, Code, Image, Link, Text } from '@chakra-ui/react'
import CodeBlock from '@/components/Blog/CodeBlock'

export default function GCG (): JSX.Element {
  return (
    <>
      <Head><title>Blog - GCG</title></Head>
      <Text fontSize={'l'} textAlign={'right'}><b>Published:</b> 01.05.2025</Text>

      <Text fontSize={'5xl'} textAlign={'center'}>
        GCG: Adversarial Attacks on Large Language Models
      </Text>
      <Center>
        <Link target="_blank" href="">
          <Image src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
        </Link>
      </Center>

      <Center mb={5} className="flex flex-col">
        <Image src="/imgs/blog/gcg/gcg.png" alt="Example of GCG messages with suffix to be optimized and target response." />
        <Text textColor={'gray.500'} fontSize={'sm'} textAlign={'center'}>Example of GCG messages with suffix to be optimized (yellow) and target response (green).</Text>
      </Center>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Introduction</Text>
      <Text mb={5}>Greedy Coordinate Gradient (<b>GCG</b>) is a technique to craft adversarial attacks on Aligned Large Language Models proposed in <Link href="">Universal and Transferable Adversarial Attacks on Aligned Language Models</Link> </Text>

      <Text mb={5}>The Colab Notebook with the shown implementation is freely accessible at <Link textColor={'blue.500'} href="">this link</Link>, while the <Link href="" textColor={'blue.500'}>GitHub repository</Link> contains .py files.</Text>
    </>
  )
}

GCG.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
      <BlogLayout>{page}</BlogLayout>
    </AppLayout>
  )
}
