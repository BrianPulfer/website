import Link from "next/link";
import { Box, GridItem, SimpleGrid, IconButton, useColorMode, useColorModeValue } from "@chakra-ui/react";
import { SunIcon, MoonIcon } from "@chakra-ui/icons";

export default function Navbar() {
  const { colorMode, toggleColorMode } = useColorMode();

  const paths = {
    "/": "Brian Pulfer",
    "/projects": "Projects",
    "/blog": "Blog",
    "/publications": "Publications"
  }

  const nPaths = Object.keys(paths).length;

  let pageLinks = [];
  for (const [path, title] of Object.entries(paths)) {
    pageLinks.push(
      <GridItem key={path} textAlign={"center"} className={"hover:text-gray-400"}>
          <Link href={path}>
            <Box fontSize="xl" fontWeight="bold" >
              {title}
            </Box>
          </Link>
        </GridItem>
    )
  }

  return (
    <Box
      shadow="md"
      py={1}
      position="sticky"
      top="0"
      zIndex="999"
      className={"mb-10"}
      bgColor={useColorModeValue("#EEE", "#111")}
    >
      <SimpleGrid justifyContent="space-between" columns={nPaths+1}>

        {pageLinks}
        
        <GridItem textAlign={"right"}>
          <IconButton
            aria-label="Toggle color mode"
            icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
            onClick={toggleColorMode}
          />
        </GridItem>
      </SimpleGrid>
    </Box>
  );
}