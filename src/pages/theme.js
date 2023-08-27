import { extendTheme } from "@chakra-ui/react";

const black = "#000"
const white = "#fff"

const theme = extendTheme({
  config: {
    initialColorMode: "dark", // 'dark', 'system', 'light'
    useSystemColorMode: true,
  },
  fonts: {
    heading: `'Open Sans', sans-serif`,
    body: `'Raleway', sans-serif`,
  },

  // Custom colors
  styles: {
    global: (props) => ({
      body: {
        bg: props.colorMode === "dark" ? black : white,
        color: props.colorMode === "dark" ? white : black,
      },
    }),
  },
});

export default theme;