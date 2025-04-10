import { ThemeProvider, createTheme } from "@mui/material/styles";
import { CssBaseline } from "@mui/material";
import { ReactNode } from "react";

interface RootProps {
  children: ReactNode;
}

const theme = createTheme({
  colorSchemes: {
    light: {
      palette: {
        primary: {
          main: "#005eb8",
        },
        secondary: {
          main: "#dc004e",
        },
        background: {
          default: "#ffffff",
          paper: "#ffffff",
        },
      },
    },
    dark: {
      palette: {
        primary: {
          main: "#17dffb",
        },
        secondary: {
          main: "#f48fb1",
        },
        background: {
          default: "#121212",
          paper: "#424242",
        },
      },
    },
  },
});

export default function Root({ children }: RootProps) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
