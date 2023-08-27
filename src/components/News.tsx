import { Box, Text } from "@chakra-ui/react";
import { useColorModeValue } from "@chakra-ui/react";


export default function News({title, children}: {title: string, children: React.ReactNode}) {
  const bgColor = useColorModeValue("#f5f5f5", "#252525");

  return (
    <Box bgColor={bgColor} width={"100%"} className={"px-10 py-5"} borderRadius={"xl"} boxShadow={"2xl"} >
      <Text textAlign={"right"} fontSize={"xl"} fontWeight={"bold"}>
          {title}
      </Text>
      <Text fontSize={"l"}>
        {children}
      </Text>
    </Box>
  );
}