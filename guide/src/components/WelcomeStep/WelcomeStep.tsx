import { Box, Typography, Alert } from "@mui/material";

export default function WelcomeStep() {
  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="body1" sx={{ mb: 3 }}>
          Welcome to LLM Evaluation Guide! This interactive guide will help you
          select suitable evaluation methods for your use-case.
        </Typography>

        <Alert severity="info">
          <Typography sx={{ mb: 1, textAlign: "justify" }}>
            <strong>Note:</strong> This guide is intended to be used by users
            with basic technical knowledge about LLMs. If you are unfamiliar
            with this topic, we suggest liaising with someone more experienced
            to help you with the selection process.
          </Typography>
          <Typography sx={{ textAlign: "justify" }}>
            The guide is intended to provide helpful information on relevant
            evaluation methods, but cannot cover all possible scenarios and
            requirements. We recommend considering the details of your use-case
            and potential domain-specific evaluation methods beyond the
            recommendations provided by this guide.
          </Typography>
        </Alert>
      </Box>
    </Box>
  );
}
