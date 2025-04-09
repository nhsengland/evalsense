import React from "react";
import Modal from "@mui/material/Modal";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Link from "@mui/material/Link";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import ReactMarkdown from "react-markdown";
import { getItemById } from "@site/src/utils/dataLoaders";

// Basic styling for the modal content
const style = {
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

export default function MethodDetailsModal({ method, open, onClose }) {
  if (!method) return null;

  const categoryName = getItemById("categories", method.category).name;

  // TODO: Implement loading long description from MD file if needed
  const longDescription = method.description_long_file
    ? `*Long description content from ${method.description_long_file} would go here.*`
    : method.description_long || method.description_short; // Fallback

  return (
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="method-details-title"
      aria-describedby="method-details-description"
    >
      <Box sx={style}>
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
        {/* Use ReactMarkdown for rendering long description */}
        <ReactMarkdown>{longDescription}</ReactMarkdown>

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
                      ref.url ? (
                        <Link
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {ref.name}
                        </Link>
                      ) : (
                        ref.name
                      )
                    }
                    secondary={ref.bib_record || ""} // Optional Bib record display
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
