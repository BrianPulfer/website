import { Box } from '@chakra-ui/react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

import { useColorModeValue } from '@chakra-ui/react';


export default function CodeBlock({ language, children }) {

    return (
        <Box my={4} >
          <SyntaxHighlighter language={language} showLineNumbers={true} showInlineLineNumbers={true}>
            {children}
          </SyntaxHighlighter>
        </Box>
      );
}