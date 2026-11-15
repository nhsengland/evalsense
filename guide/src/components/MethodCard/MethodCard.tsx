import React, { useState } from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Box from "@mui/material/Box";

import MethodDetails from "@site/src/components/MethodDetails/MethodDetails";
import {
  CoverageLevel,
  Method,
  Quality,
  Risk,
} from "@site/src/types/evaluation.types";
import { getItemById } from "@site/src/utils/dataLoaders";
import CoverageInfo from "../CoverageChip/CoverageChip";

interface MethodCardProps {
  method: Method | undefined;
  onSelect?: (methodId: string) => void;
  isSelected?: boolean;
  showCoverageFor?: Quality | Risk;
  showImplementationStatus?: boolean;
}

const MethodCard: React.FC<MethodCardProps> = ({
  method,
  onSelect,
  isSelected,
  showCoverageFor,
  showImplementationStatus = false,
}) => {
  const [modalOpen, setModalOpen] = useState(false);

  if (!method) return null;

  const category = getItemById("categories", method.category);
  const categoryName = category?.name || method.category;

  let coverage: CoverageLevel | null = null;
  if (showCoverageFor) {
    const coverageList =
      showCoverageFor.type === "quality"
        ? method.assessed_qualities
        : method.identified_risks;
    coverage = coverageList.find(
      (cov) => cov.id === showCoverageFor.id,
    )?.coverage;
  }

  const handleOpenModal = () => setModalOpen(true);
  const handleCloseModal = () => setModalOpen(false);

  return (
    <>
      <Card sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <CardContent sx={{ flexGrow: 1 }}>
          <Typography gutterBottom variant="h6" component="h2">
            {method.name || "Unnamed Method"}
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 1 }}>
            <Chip label={categoryName} size="small" />
            {showImplementationStatus && (
              <Chip
                label={
                  method.link_implementation
                    ? "Implemented in EvalSense"
                    : "Not yet implemented"
                }
                color={method.link_implementation ? "success" : "default"}
                size="small"
                variant="outlined"
              />
            )}
          </Box>
          <Typography variant="body2" color="text.secondary">
            {method.description_short || "No description."}
          </Typography>
        </CardContent>
        {coverage && (
          <CardContent sx={{ mt: 0, mb: 0 }}>
            <CoverageInfo coverageLevel={coverage} size="small" />
          </CardContent>
        )}
        <CardActions>
          <Button size="small" onClick={handleOpenModal}>
            Details
          </Button>
          {onSelect && (
            <Button
              size="small"
              onClick={() => onSelect(method.id)}
              variant={isSelected ? "contained" : "outlined"}
            >
              {isSelected ? "Selected" : "Select"}
            </Button>
          )}
        </CardActions>
      </Card>
      <MethodDetails
        method={method}
        open={modalOpen}
        onClose={handleCloseModal}
      />
    </>
  );
};

export default MethodCard;
