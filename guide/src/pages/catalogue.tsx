import { useState, useMemo } from "react";
import Layout from "@theme/Layout";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Alert from "@mui/material/Alert";

import CatalogueFilter from "@site/src/components/CatalogueFilter/CatalogueFilter";
import MethodCard from "@site/src/components/MethodCard/MethodCard";
import { getData } from "@site/src/utils/dataLoaders";
import { CatalogueFilters } from "../types/evaluation.types";

export default function CataloguePage() {
  const [filters, setFilters] = useState<CatalogueFilters>({});

  const filteredMethods = useMemo(() => {
    const {
      searchText,
      tasks = [],
      qualities = [],
      risks = [],
      categories = [],
    } = filters;

    return getData("methods").filter((method) => {
      // Text search
      if (searchText) {
        const lowerSearch = searchText.toLowerCase();
        if (
          !method.name.toLowerCase().includes(lowerSearch) &&
          !method.description_short.toLowerCase().includes(lowerSearch)
        ) {
          return false;
        }
      }

      // Task filter
      if (
        tasks.length > 0 &&
        !tasks.some((taskId) => method.supported_tasks.includes(taskId))
      ) {
        return false;
      }

      // Quality filter
      if (
        qualities.length > 0 &&
        !qualities.some((qualId) =>
          method.assessed_qualities.some((q) => q.id === qualId),
        )
      ) {
        return false;
      }

      // Risk filter
      if (
        risks.length > 0 &&
        !risks.some((riskId) =>
          method.identified_risks.some((r) => r.id === riskId),
        )
      ) {
        return false;
      }

      // Category filter
      if (categories.length > 0 && !categories.includes(method.category)) {
        return false;
      }

      return true;
    });
  }, [filters]);

  return (
    <Layout title="Method Catalogue" description="Browse Evaluation Methods">
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Evaluation Method Catalogue
        </Typography>

        <CatalogueFilter filters={filters} onFilterChange={setFilters} />

        {filteredMethods.length > 0 ? (
          <Grid container spacing={3}>
            {filteredMethods.map((method) => (
              <Grid key={method.id} size={{ xs: 12, sm: 6, md: 4 }}>
                <MethodCard method={method} />{" "}
              </Grid>
            ))}
          </Grid>
        ) : (
          <Alert severity="info">
            No evaluation methods match the current filters.
          </Alert>
        )}
      </Container>
    </Layout>
  );
}
