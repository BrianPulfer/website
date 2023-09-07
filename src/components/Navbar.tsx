import { useState } from "react";
import Link from "next/link";
import {
  Box,
  Text,
  Flex,
  IconButton,
  useColorMode,
  useColorModeValue,
} from "@chakra-ui/react";
import { SunIcon, MoonIcon, HamburgerIcon, CloseIcon } from "@chakra-ui/icons";

// TODO: Collapse navbar past a certain width
export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const { colorMode, toggleColorMode } = useColorMode();

  const paths = {
    "/": "Brian Pulfer",
    "/research": "Research",
    "/projects": "Projects",
    "/blog": "Blog",
  };

  const nPaths = Object.keys(paths).length;

  let pageLinks = [];
  for (const [path, title] of Object.entries(paths)) {
    pageLinks.push(
      <Box key={path} textAlign={"center"}>
        <Link href={path} onClick={() => setIsOpen(false)}>
          <Text
            fontSize="xl"
            fontWeight="bold"
            className={"hover:text-gray-400"}
          >
            {title}
          </Text>
        </Link>
      </Box>
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
      <Box
        display={{ base: "block", md: "none" }}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? (
          <CloseIcon boxSize={5} _hover={{ color: "gray.500" }} />
        ) : (
          <HamburgerIcon boxSize={6} _hover={{ color: "gray.500" }} />
        )}
      </Box>
      <Flex
        display={{ base: isOpen ? "flex" : "none", md: "flex" }}
        justifyContent="space-between"
        flexBasis={{ base: "100%", md: "auto" }}
        flexDir={{ base: "column", md: "row" }}
        alignItems="center"
      >
        {pageLinks}

        <Box textAlign={"center"}>
          <IconButton
            aria-label="Toggle color mode"
            icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
            onClick={toggleColorMode}
          />
        </Box>
      </Flex>
    </Box>
  );
}
