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
          <strong>Note:</strong> This LLM evaluation guide is intended to be
          used by users with basic technical knowledge about LLMs. If you are
          unfamiliar with this topic, we suggest liaising with someone more
          experienced to help you with the selection process.
        </Alert>
      </Box>
    </Box>
  );
}
