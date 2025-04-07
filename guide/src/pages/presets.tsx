import React from "react";
import Layout from "@theme/Layout";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Button from "@mui/material/Button";
import presetData from "@site/src/data/presets.json";
import { setPresetToLoad } from "@site/src/utils/localStorage";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function PresetsPage() {
  const guideUrl = useBaseUrl("/guide"); // Get correct base URL for navigation

  const handleLoadPreset = (preset) => {
    if (preset.guideState) {
      setPresetToLoad(preset.guideState); // Store the preset's state
      window.location.href = guideUrl; // Navigate to the guide page
    } else {
      console.error("Preset is missing guideState:", preset.id);
    }
  };

  return (
    <Layout title="Presets" description="Preconfigured Evaluation Scenarios">
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Evaluation Presets
        </Typography>
        <Typography variant="body1" gutterBottom sx={{ mb: 3 }}>
          Select a common scenario to pre-fill the interactive guide with
          relevant settings.
        </Typography>

        <Grid container spacing={3}>
          {presetData.map((preset) => (
            <Grid key={preset.id} size={{ xs: 12, sm: 6, md: 4 }}>
              <Card
                sx={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6">{preset.name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {preset.description}
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button size="small" onClick={() => handleLoadPreset(preset)}>
                    Load Preset
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Layout>
  );
}
