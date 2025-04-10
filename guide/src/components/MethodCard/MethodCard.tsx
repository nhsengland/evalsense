import React, { useState } from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";

import MethodDetails from "@site/src/components/MethodDetails/MethodDetails";
import { Method, Quality, Risk } from "@site/src/types/evaluation.types";
import { getItemById } from "@site/src/utils/dataLoaders";

interface MethodCardProps {
  method: Method | undefined;
  onSelect?: (methodId: string) => void;
  isSelected?: boolean;
  showCoverageFor?: Quality | Risk;
}

const MethodCard: React.FC<MethodCardProps> = ({
  method,
  onSelect,
  isSelected,
  showCoverageFor,
}) => {
  const [modalOpen, setModalOpen] = useState(false);

  if (!method) return null;

  const category = getItemById("categories", method.category);
  const categoryName = category?.name || method.category;

  let coverageInfo: string | null = null;
  if (showCoverageFor) {
    const coverageList =
      showCoverageFor.type === "quality"
        ? method.assessed_qualities
        : method.identified_risks;
    const coverage = coverageList.find(
      (cov) => cov.id === showCoverageFor.id
    )?.coverage;
    if (coverage) {
      coverageInfo = `Coverage: ${coverage}`;
    }
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
          <Chip label={categoryName} size="small" sx={{ mb: 1 }} />
          <Typography variant="body2" color="text.secondary">
            {method.description_short || "No description."}
          </Typography>
          {coverageInfo && (
            <Typography
              variant="caption"
              color="primary"
              display="block"
              sx={{ mt: 1 }}
            >
              {coverageInfo}
            </Typography>
          )}
        </CardContent>
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
