import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";
import Alert from "@mui/material/Alert";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import WarningIcon from "@mui/icons-material/Warning";
import CancelIcon from "@mui/icons-material/Cancel";
import { getItemById, getItemsByIds } from "@site/src/utils/dataLoaders";
import { calculateCoverage } from "@site/src/utils/evaluationLogic";
import {
  GuideAnswers,
  ImportanceRating,
} from "@site/src/types/evaluation.types";
import CoverageChip from "../CoverageChip/CoverageChip";

interface SummaryReportProps {
  answers: GuideAnswers;
  selectedMethodIds: string[];
}

export default function SummaryReport({
  answers,
  selectedMethodIds,
}: SummaryReportProps) {
  const task = getItemById("tasks", answers.q_task_type as string);

  // Get quality and risk ratings
  const qualityRatings =
    (answers.q_qualities as ImportanceRating[] | undefined) || [];
  const riskRatings = (answers.q_risks as ImportanceRating[] | undefined) || [];

  // Get IDs for qualities and risks with importance > 1
  const desiredQualityIds = qualityRatings
    .filter((q) => q.importance > 1)
    .map((q) => q.id);
  const desiredRiskIds = riskRatings
    .filter((r) => r.importance > 1)
    .map((r) => r.id);

  const desiredQualities = getItemsByIds("qualities", desiredQualityIds);
  const desiredRisks = getItemsByIds("risks", desiredRiskIds);
  const selectedMethods = getItemsByIds("methods", selectedMethodIds);

  const { coverage, uncovered, partiallyCovered } = calculateCoverage(
    selectedMethodIds,
    desiredQualities,
    desiredRisks,
    qualityRatings,
    riskRatings,
  );
  console.log(partiallyCovered);

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
                <ListItemText primary={`• ${method.name}`} />
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
              coverage[q.id] == "Good" || coverage[q.id] == "Very Good" ? (
                <CheckCircleIcon color="success" sx={{ mr: 1 }} />
              ) : (
                <WarningIcon color="warning" sx={{ mr: 1 }} />
              )
            ) : (
              <CancelIcon color="error" sx={{ mr: 1 }} />
            )}
            <Typography component="span">Quality: {q.name}</Typography>
            <CoverageChip
              coverageLevel={coverage[q.id]}
              size="small"
              sx={{ ml: 1 }}
            />
          </Box>
        ))}
        {desiredRisks.map((r) => (
          <Box key={r.id} display="flex" alignItems="center" mb={1}>
            {coverage[r.id] ? (
              coverage[r.id] == "Good" || coverage[r.id] == "Very Good" ? (
                <CheckCircleIcon color="success" sx={{ mr: 1 }} />
              ) : (
                <WarningIcon color="warning" sx={{ mr: 1 }} />
              )
            ) : (
              <CancelIcon color="error" sx={{ mr: 1 }} />
            )}
            <Typography component="span">Risk: {r.name}</Typography>
            <CoverageChip
              coverageLevel={coverage[r.id]}
              size="small"
              sx={{ ml: 1 }}
            />
          </Box>
        ))}
      </Box>

      {(uncovered.qualities.length > 0 ||
        uncovered.risks.length > 0 ||
        partiallyCovered.qualities.length > 0 ||
        partiallyCovered.risks.length > 0) && (
        <Alert severity="warning">
          <strong>Attention</strong>
          {(uncovered.qualities.length > 0 || uncovered.risks.length > 0) && (
            <>
              <br />
              <br />
              The following requirements are not covered by your selected
              methods:
              <List dense sx={{ listStyleType: "disc" }}>
                {uncovered.qualities.map((q) => (
                  <ListItemText key={q.id} primary={`• Quality: ${q.name}`} />
                ))}
                {uncovered.risks.map((r) => (
                  <ListItemText key={r.id} primary={`• Risk: ${r.name}`} />
                ))}
              </List>
              Consider using additional methods to achieve better coverage.
            </>
          )}
          {(partiallyCovered.qualities.length > 0 ||
            partiallyCovered.risks.length > 0) && (
            <>
              <br />
              <br />
              The following requirements are partially covered by your selected
              methods:
              <List dense>
                {partiallyCovered.qualities.map((q) => (
                  <ListItemText key={q.id} primary={`• Quality: ${q.name}`} />
                ))}
                {partiallyCovered.risks.map((r) => (
                  <ListItemText key={r.id} primary={`• Risk: ${r.name}`} />
                ))}
              </List>
              Consider choosing methods with better coverage for these
              requirements.
            </>
          )}
        </Alert>
      )}

      {selectedMethods.length > 0 &&
        uncovered.qualities.length === 0 &&
        uncovered.risks.length === 0 &&
        partiallyCovered.qualities.length === 0 &&
        partiallyCovered.risks.length === 0 && (
          <Alert severity="success">
            All specified qualities and risks appear to be addressed by your
            selected methods based on available data. Remember to consult method
            details for nuances.
          </Alert>
        )}
    </Paper>
  );
}
