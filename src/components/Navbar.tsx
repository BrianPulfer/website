import Link from "next/link";
import {
  Box,
  Text,
  GridItem,
  SimpleGrid,
  IconButton,
  useColorMode,
  useColorModeValue,
} from "@chakra-ui/react";
import { SunIcon, MoonIcon } from "@chakra-ui/icons";

export default function Navbar() {
  const { colorMode, toggleColorMode } = useColorMode();

  const paths = {
    "/": "Brian Pulfer",
    "/publications": "Publications",
    "/projects": "Projects",
    "/blog": "Blog",
  };

  const nPaths = Object.keys(paths).length;

  let pageLinks = [];
  for (const [path, title] of Object.entries(paths)) {
    pageLinks.push(
      <GridItem key={path} textAlign={"center"}>
        <Link href={path}>
          <Text
            fontSize="xl"
            fontWeight="bold"
            className={"hover:text-gray-400"}
          >
            {title}
          </Text>
        </Link>
      </GridItem>
    );
  }

  return (
    <Box
      shadow="xl"
      py={4}
      px={6}
      position="sticky"
      top="0"
      zIndex="999"
      bgColor={useColorModeValue("#EEE", "#111")}
      className={"mb-10"}
    >
      <SimpleGrid justifyContent="space-between" columns={nPaths + 1}>
        {pageLinks}

        <GridItem textAlign={"center"}>
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
