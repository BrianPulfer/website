import { Box } from '@chakra-ui/react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'

export default function CodeBlock ({ language, children }: { language: any, children: any }): JSX.Element {
  return (
        <Box my={4} >
          <SyntaxHighlighter language={language} showLineNumbers={true} showInlineLineNumbers={true}>
            {children}
          </SyntaxHighlighter>
        </Box>
  )
}
