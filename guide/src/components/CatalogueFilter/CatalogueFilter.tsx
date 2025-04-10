import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import Checkbox from "@mui/material/Checkbox";
import { getData } from "@site/src/utils/dataLoaders";
import { CatalogueFilters } from "@site/src/types/evaluation.types";

interface CatalogueFilterProps {
  filters: CatalogueFilters;
  onFilterChange: (filters: CatalogueFilters) => void;
}

// Load data for filter options
const taskOptions = getData("tasks");
const qualityOptions = getData("qualities");
const riskOptions = getData("risks");
const categoryOptions = getData("categories");

export default function CatalogueFilter({
  filters,
  onFilterChange,
}: CatalogueFilterProps) {
  const handleAutocompleteChange = (field) => (event, newValue) => {
    // newValue is an array of selected option objects
    onFilterChange({
      ...filters,
      [field]: newValue.map((option) => option.id),
    });
  };

  const handleTextChange = (event) => {
    onFilterChange({ ...filters, searchText: event.target.value });
  };

  // Helper to get the current value array for Autocomplete based on IDs
  const getAutocompleteValue = (field, options) => {
    const ids = filters[field] || [];
    return options.filter((option) => ids.includes(option.id));
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        mb: 3,
        p: 2,
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          gap: 2,
          justifyContent: "center",
        }}
      >
        <TextField
          label="Search Methods"
          variant="outlined"
          size="small"
          value={filters.searchText || ""}
          onChange={handleTextChange}
          sx={{ width: "65%" }}
        />
        <Autocomplete
          multiple
          size="small"
          limitTags={1}
          options={taskOptions}
          getOptionLabel={(option) => option.name}
          value={getAutocompleteValue("tasks", taskOptions)}
          onChange={handleAutocompleteChange("tasks")}
          disableCloseOnSelect
          renderOption={(props, option, { selected }) => (
            <li {...props}>
              <Checkbox style={{ marginRight: 8 }} checked={selected} />
              {option.name}
            </li>
          )}
          renderInput={(params) => (
            <TextField {...params} label="Tasks" placeholder="Filter by Task" />
          )}
          sx={{ width: "32%" }}
        />
      </Box>

      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          gap: 2,
          justifyContent: "center",
        }}
      >
        <Autocomplete
          multiple
          size="small"
          limitTags={1}
          options={qualityOptions}
          getOptionLabel={(option) => option.name}
          value={getAutocompleteValue("qualities", qualityOptions)}
          onChange={handleAutocompleteChange("qualities")}
          disableCloseOnSelect
          renderOption={(props, option, { selected }) => (
            <li {...props}>
              <Checkbox style={{ marginRight: 8 }} checked={selected} />
              {option.name}
            </li>
          )}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Qualities"
              placeholder="Filter by Quality"
            />
          )}
          sx={{ width: "32%" }}
        />
        <Autocomplete
          multiple
          size="small"
          limitTags={1}
          options={riskOptions}
          getOptionLabel={(option) => option.name}
          value={getAutocompleteValue("risks", riskOptions)}
          onChange={handleAutocompleteChange("risks")}
          disableCloseOnSelect
          renderOption={(props, option, { selected }) => (
            <li {...props}>
              <Checkbox style={{ marginRight: 8 }} checked={selected} />
              {option.name}
            </li>
          )}
          renderInput={(params) => (
            <TextField {...params} label="Risks" placeholder="Filter by Risk" />
          )}
          sx={{ width: "32%" }}
        />
        <Autocomplete
          multiple
          size="small"
          limitTags={1}
          options={categoryOptions}
          getOptionLabel={(option) => option.name}
          value={getAutocompleteValue("categories", categoryOptions)}
          onChange={handleAutocompleteChange("categories")}
          disableCloseOnSelect
          renderOption={(props, option, { selected }) => (
            <li {...props}>
              <Checkbox style={{ marginRight: 8 }} checked={selected} />
              {option.name}
            </li>
          )}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Categories"
              placeholder="Filter by Category"
            />
          )}
          sx={{ width: "32%" }}
        />
      </Box>
    </Box>
  );
}
