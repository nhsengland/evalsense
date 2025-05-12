import Modal from "@mui/material/Modal";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Link from "@mui/material/Link";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import CloseIcon from "@mui/icons-material/Close";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import Tooltip from "@mui/material/Tooltip";
import CircularProgress from "@mui/material/CircularProgress";
import { getItemById } from "@site/src/utils/dataLoaders";
import { Method, CoverageLevel } from "@site/src/types/evaluation.types";
import { useState, useEffect } from "react";

// Define interface for component props
interface MethodDetailsModalProps {
  method: Method | undefined;
  open: boolean;
  onClose: () => void;
}

// Basic styling for the modal content
const modalStyle = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: "80%",
  maxWidth: 800,
  bgcolor: "background.paper",
  border: "1px solid #ccc",
  boxShadow: 24,
  p: 4,
  maxHeight: "90vh",
  overflowY: "auto",
};

// Coverage level colors
const coverageColors: Record<CoverageLevel, string> = {
  "Very Good": "#4caf50", // green
  Good: "#8bc34a", // lightGreen
  Partial: "#ffeb3b", // yellow
  Poor: "#ff9800", // orange
};

export default function MethodDetailsModal({
  method,
  open,
  onClose,
}: MethodDetailsModalProps) {
  const [LongDescription, setLongDescription] = useState<React.FC | null>(null);
  const [hasLongDescription, setHasLongDescription] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    if (!method) {
      return;
    }

    setIsLoading(true);

    async function loadLongDescription(fileName: string) {
      try {
        const { default: Description } = await import(
          `@site/src/data/descriptions/${fileName}`
        );
        setLongDescription(() => Description);
        setHasLongDescription(true);
      } catch (error) {
        console.error(`Failed to load description file: ${fileName}`, error);
      }
      setIsLoading(false);
    }

    if (method.description_long_file) {
      loadLongDescription(method.description_long_file);
    } else {
      setIsLoading(false);
    }
  }, [method]);

  // Function to copy bib record to clipboard
  const handleCopyBibRecord = (bibRecord: string, refId: string) => {
    navigator.clipboard
      .writeText(bibRecord)
      .then(() => {
        setCopiedId(refId);
        // Reset the copied state after 2 seconds
        setTimeout(() => setCopiedId(null), 2000);
      })
      .catch((err) => {
        console.error("Failed to copy text: ", err);
      });
  };

  if (!method) return null;

  const categoryName = getItemById("categories", method.category).name;

  return (
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="method-details-title"
      aria-describedby="method-details-description"
    >
      <Box sx={modalStyle}>
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: "absolute",
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>

        <Typography id="method-details-title" variant="h5" component="h2">
          {method.name}
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          Category: {categoryName}
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="h6">Description</Typography>
        {hasLongDescription || isLoading ? (
          isLoading ? (
            <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
              <CircularProgress />
            </Box>
          ) : (
            hasLongDescription && <LongDescription />
          )
        ) : (
          <Typography variant="body2">{method.description_short}</Typography>
        )}

        <Divider sx={{ my: 2 }} />

        <Typography variant="h6">Assessed Qualities</Typography>
        {method.assessed_qualities.length > 0 ? (
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1, mb: 2 }}>
            {method.assessed_qualities.map((quality) => {
              const qualityItem = getItemById("qualities", quality.id);
              return (
                <Tooltip
                  key={quality.id}
                  title={qualityItem.description || qualityItem.name}
                >
                  <Chip
                    label={`${qualityItem.name} (${quality.coverage})`}
                    sx={{
                      backgroundColor: coverageColors[quality.coverage],
                      color: ["Very Good"].includes(quality.coverage)
                        ? "white"
                        : "black",
                    }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        ) : (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 1, mb: 2 }}
          >
            This method does not assess any qualities.
          </Typography>
        )}

        <Typography variant="h6">Covered Risks</Typography>
        {method.identified_risks.length > 0 ? (
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1, mb: 2 }}>
            {method.identified_risks.map((risk) => {
              const riskItem = getItemById("risks", risk.id);
              return (
                <Tooltip
                  key={risk.id}
                  title={riskItem.description || riskItem.name}
                >
                  <Chip
                    label={`${riskItem.name} (${risk.coverage})`}
                    sx={{
                      backgroundColor: coverageColors[risk.coverage],
                      color: ["Very Good"].includes(risk.coverage)
                        ? "white"
                        : "black",
                    }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        ) : (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 1, mb: 2 }}
          >
            This method does not cover any risks.
          </Typography>
        )}

        <Divider sx={{ my: 2 }} />

        {method.output_values && (
          <>
            <Typography variant="h6">Output Values</Typography>
            <Typography variant="body2">{method.output_values}</Typography>
            <Divider sx={{ my: 2 }} />
          </>
        )}

        {method.link_implementation && (
          <>
            <Typography variant="h6">Implementation</Typography>
            <Link
              href={method.link_implementation}
              target="_blank"
              rel="noopener noreferrer"
            >
              {method.link_implementation}
            </Link>
            <Divider sx={{ my: 2 }} />
          </>
        )}

        <Box display="flex" gap={3} my={2}>
          <Box flex={1}>
            <Typography variant="h6">Advantages</Typography>
            <List dense>
              {method.advantages?.map((adv, index) => (
                <ListItem key={index} disablePadding>
                  <ListItemText primary={`• ${adv}`} />
                </ListItem>
              ))}
            </List>
          </Box>
          <Box flex={1}>
            <Typography variant="h6">Disadvantages</Typography>
            <List dense>
              {method.disadvantages?.map((dis, index) => (
                <ListItem key={index} disablePadding>
                  <ListItemText primary={`• ${dis}`} />
                </ListItem>
              ))}
            </List>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {method.references?.length > 0 && (
          <>
            <Typography variant="h6">References</Typography>
            <List dense>
              {method.references.map((ref, index) => (
                <ListItem key={index} disablePadding>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center">
                        {ref.url ? (
                          <Link
                            href={ref.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {ref.name}
                          </Link>
                        ) : (
                          ref.name
                        )}
                        {ref.bib_record && (
                          <Tooltip
                            title={
                              copiedId === `ref-${index}`
                                ? "Copied!"
                                : "Copy BibTeX"
                            }
                          >
                            <IconButton
                              size="small"
                              onClick={() =>
                                handleCopyBibRecord(
                                  ref.bib_record,
                                  `ref-${index}`,
                                )
                              }
                              sx={{ ml: 1 }}
                              color={
                                copiedId === `ref-${index}`
                                  ? "success"
                                  : "default"
                              }
                            >
                              <ContentCopyIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        )}
      </Box>
    </Modal>
  );
}
