import Chip from "@mui/material/Chip";
import { CoverageLevel } from "@site/src/types/evaluation.types";
import { SxProps, Theme } from "@mui/material/styles";

interface CoverageLevelProps {
  coverageLevel: CoverageLevel | null;
  size?: "small" | "medium";
  sx?: SxProps<Theme>;
}

const coverageColors: Record<CoverageLevel | "No", string> = {
  "Very Good": "#4caf50",
  Good: "#8bc34a",
  Partial: "#ffeb3b",
  Poor: "#ff9800",
  No: "#ef5350",
};

export default function CoverageChip({
  coverageLevel,
  size,
  sx,
}: CoverageLevelProps) {
  let coverageDescription: string;
  if (!coverageLevel) {
    coverageDescription = "No coverage";
  } else {
    const normalisedCoverageLevel =
      coverageLevel.charAt(0).toUpperCase() +
      coverageLevel.slice(1).toLowerCase();
    coverageDescription = `Coverage: ${normalisedCoverageLevel}`;
  }

  return (
    <Chip
      label={coverageDescription}
      size={size}
      sx={{
        backgroundColor: coverageColors[coverageLevel || "No"],
        color:
          coverageLevel == null || coverageLevel == "Very Good"
            ? "white"
            : "black",
        ...sx,
      }}
    />
  );
}
