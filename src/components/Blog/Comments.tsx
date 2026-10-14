import Giscus from '@giscus/react'
import { Box, Link, Text, useColorMode } from '@chakra-ui/react'

export default function Comments (): JSX.Element {
  const { colorMode } = useColorMode()

  return (
    <Box mt={16} w={'100%'}>
      <Text fontSize={'3xl'} fontWeight={'bold'} mb={2} textAlign={'center'}>
        Comments 💬
      </Text>
      <Text fontSize={'sm'} textColor={'gray.500'} mb={5} textAlign={'center'}>
        Sign in with <Link textColor={'blue.500'} href={'https://github.com/signup'} isExternal>GitHub</Link> to
        comment below or leave a reaction (👍 ❤️ 🚀) as a like on the post.
      </Text>
      <Giscus
        id="comments"
        repo="BrianPulfer/website"
        repoId="MDEwOlJlcG9zaXRvcnkyNTc2NDU2NDg="
        category="Announcements"
        categoryId="DIC_kwDOD1tcUM4DIYWv"
        mapping="pathname"
        strict="0"
        reactionsEnabled="1"
        emitMetadata="0"
        inputPosition="bottom"
        theme={colorMode === 'light' ? 'light' : 'dark'}
        lang="en"
        loading="lazy"
      />
    </Box>
  )
}
