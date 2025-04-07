import React from "react";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import { getItemById, getItemsByIds } from "@site/src/utils/dataLoaders";
import { calculateCoverage } from "@site/src/utils/evaluationLogic";

export default function SummaryReport({ answers, selectedMethodIds }) {
  const task = getItemById("tasks", answers.q_task_type);
  const desiredQualities = getItemsByIds(
    "qualities",
    answers.q_qualities || []
  );
  const desiredRisks = getItemsByIds("risks", answers.q_risks || []);
  const selectedMethods = getItemsByIds("methods", selectedMethodIds);

  const { coverage, uncovered } = calculateCoverage(
    selectedMethodIds,
    desiredQualities,
    desiredRisks
  );

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Evaluation Plan Summary
      </Typography>

      <Box mb={3}>
        <Typography variant="h6">Your Requirements:</Typography>
        <List dense disablePadding>
          <ListItem>
            <ListItemText
              primary="Task"
              secondary={task?.name || "Not specified"}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Desired Qualities"
              secondary={
                desiredQualities.map((q) => q.name).join(", ") || "None"
              }
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Desired Risks to Mitigate"
              secondary={desiredRisks.map((r) => r.name).join(", ") || "None"}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Reference Data Available"
              secondary={answers.q_references === "yes" ? "Yes" : "No"}
            />
          </ListItem>
        </List>
      </Box>

      <Divider sx={{ my: 2 }} />

      <Box mb={3}>
        <Typography variant="h6">Selected Evaluation Methods:</Typography>
        {selectedMethods.length > 0 ? (
          <List dense disablePadding>
            {selectedMethods.map((method) => (
              <ListItem key={method.id}>
                <ListItemText primary={method.name} />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No methods selected.
          </Typography>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      <Box mb={3}>
        <Typography variant="h6">Coverage Analysis:</Typography>
        {desiredQualities.map((q) => (
          <Box key={q.id} display="flex" alignItems="center" mb={1}>
            {coverage[q.id] ? (
              <CheckCircleIcon color="success" sx={{ mr: 1 }} />
            ) : (
              <CancelIcon color="error" sx={{ mr: 1 }} />
            )}
            <Typography component="span">Quality: {q.name}</Typography>
            {coverage[q.id] && (
              <Chip
                label={`Covered (${coverage[q.id]})`}
                size="small"
                color="success"
                variant="outlined"
                sx={{ ml: 1 }}
              />
            )}
          </Box>
        ))}
        {desiredRisks.map((r) => (
          <Box key={r.id} display="flex" alignItems="center" mb={1}>
            {coverage[r.id] ? (
              <CheckCircleIcon color="success" sx={{ mr: 1 }} />
            ) : (
              <CancelIcon color="error" sx={{ mr: 1 }} />
            )}
            <Typography component="span">Risk: {r.name}</Typography>
            {coverage[r.id] && (
              <Chip
                label={`Covered (${coverage[r.id]})`}
                size="small"
                color="success"
                variant="outlined"
                sx={{ ml: 1 }}
              />
            )}
          </Box>
        ))}
      </Box>

      {(uncovered.qualities.length > 0 || uncovered.risks.length > 0) && (
        <Alert severity="warning">
          <strong>Attention:</strong> The following requirements are not fully
          covered by your selected methods:
          <List dense>
            {uncovered.qualities.map((q) => (
              <ListItemText key={q.id} primary={`Quality: ${q.name}`} />
            ))}
            {uncovered.risks.map((r) => (
              <ListItemText key={r.id} primary={`Risk: ${r.name}`} />
            ))}
          </List>
          Consider adding more methods or reviewing suggestions in the
          catalogue.
        </Alert>
      )}

      {selectedMethods.length > 0 &&
        uncovered.qualities.length === 0 &&
        uncovered.risks.length === 0 && (
          <Alert severity="success">
            All specified qualities and risks appear to be addressed by your
            selected methods based on available data. Remember to consult method
            details for nuances.
          </Alert>
        )}
    </Paper>
  );
}
