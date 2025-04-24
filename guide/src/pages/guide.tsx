import Layout from "@theme/Layout";
import InteractiveGuide from "@site/src/components/InteractiveGuide/InteractiveGuide";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";

export default function GuidePage() {
  return (
    <Layout
      title="Evaluation Guide"
      description="Interactive LLM Evaluation Guide"
    >
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          LLM Evaluation Guide
        </Typography>
        <InteractiveGuide />
      </Container>
    </Layout>
  );
}
