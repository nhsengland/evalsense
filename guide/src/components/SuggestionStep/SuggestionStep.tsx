import React from "react";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Alert from "@mui/material/Alert";

import MethodCard from "@site/src/components/MethodCard/MethodCard";
import { Method, Quality, Risk } from "@site/src/types/evaluation.types";

interface SuggestionStepProps {
  suggestedMethods: Method[];
  desiredQualities: Quality[];
  desiredRisks: Risk[];
  selectedMethodIds: string[];
  onSelectMethod: (methodId: string) => void;
}

const SuggestionStep: React.FC<SuggestionStepProps> = ({
  suggestedMethods,
  desiredQualities,
  desiredRisks,
  selectedMethodIds,
  onSelectMethod,
}) => {
  // Group methods by the desired qualities/risks they cover
  const groupedMethods: Record<
    string,
    { item: Quality | Risk; type: "quality" | "risk"; methods: Method[] }
  > = {};
  desiredQualities.forEach((q) => {
    groupedMethods[`quality_${q.id}`] = {
      item: q,
      type: "quality",
      methods: suggestedMethods.filter((m) =>
        m.assessed_qualities.some((aq) => aq.id === q.id)
      ),
    };
  });
  desiredRisks.forEach((r) => {
    groupedMethods[`risk_${r.id}`] = {
      item: r,
      type: "risk",
      methods: suggestedMethods.filter((m) =>
        m.identified_risks.some((ir) => ir.id === r.id)
      ),
    };
  });

  const hasSuggestions = Object.values(groupedMethods).some(
    (group) => group.methods.length > 0
  );

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Suggested Evaluation Methods
      </Typography>
      <Typography variant="body1" gutterBottom>
        Based on your requirements, here are some suggested methods, grouped by
        the qualities and risks you selected. The methods are ranked by how well
        they cover your overall selections. Select the methods you plan to use.
      </Typography>

      {!hasSuggestions && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          No specific methods found matching all your criteria. Consider
          broadening your requirements or exploring the full catalogue.
        </Alert>
      )}

      {Object.entries(groupedMethods).map(([groupId, group]) => {
        if (group.methods.length === 0) return null;
        return (
          <Accordion key={groupId} defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>
                For {group.type === "quality" ? "Quality" : "Risk"}:{" "}
                <strong>{group.item.name}</strong> ({group.methods.length}{" "}
                suggestions)
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {group.methods.map((method) => (
                  <Grid
                    key={`${groupId}-${method.id}`}
                    size={{ xs: 12, sm: 6, md: 4 }}
                  >
                    <MethodCard
                      method={method}
                      onSelect={onSelectMethod}
                      isSelected={selectedMethodIds.includes(method.id)}
                      showCoverageFor={group.item}
                    />
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
};
export default SuggestionStep;
