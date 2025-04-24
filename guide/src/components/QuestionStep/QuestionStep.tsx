import React, { useEffect } from "react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormGroup from "@mui/material/FormGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Rating from "@mui/material/Rating";
import Typography from "@mui/material/Typography";
import StarsIcon from "@mui/icons-material/Stars";
import SignalCellular0Bar from "@mui/icons-material/SignalCellular0Bar";
import SignalCellular1Bar from "@mui/icons-material/SignalCellular1Bar";
import SignalCellular2Bar from "@mui/icons-material/SignalCellular2Bar";
import SignalCellular3Bar from "@mui/icons-material/SignalCellular3Bar";
import SignalCellular4Bar from "@mui/icons-material/SignalCellular4Bar";
import { getItemById } from "@site/src/utils/dataLoaders";
import { getQualitiesForRisks } from "@site/src/utils/evaluationLogic";
import {
  Question,
  GuideAnswers,
  ImportanceRating,
} from "@site/src/types/evaluation.types";

// Custom icons for importance rating
const customIcons = {
  1: { icon: <SignalCellular0Bar />, label: "Not important" },
  2: { icon: <SignalCellular1Bar />, label: "Slightly important" },
  3: { icon: <SignalCellular2Bar />, label: "Moderately important" },
  4: { icon: <SignalCellular3Bar />, label: "Important" },
  5: { icon: <SignalCellular4Bar />, label: "Very important" },
};

interface IconContainerProps {
  value: number;
  [key: string]: unknown;
}

function IconContainer(props: IconContainerProps) {
  const { value, ...other } = props;
  return <span {...other}>{customIcons[value].icon}</span>;
}

interface QuestionStepProps {
  questionConfig: Question;
  currentAnswer: GuideAnswers[string];
  onChange: (answer: string | string[] | ImportanceRating[]) => void;
  allAnswers?: GuideAnswers;
}

const QuestionStep: React.FC<QuestionStepProps> = ({
  questionConfig,
  currentAnswer,
  onChange,
  allAnswers = {},
}) => {
  // For the qualities step, pre-select qualities based on selected risks
  useEffect(() => {
    if (!questionConfig) return;

    // Only proceed if this is the qualities question, there is no current selection
    // and this is the first time we are on this step
    if (
      questionConfig.id === "q_qualities" &&
      (!currentAnswer ||
        (Array.isArray(currentAnswer) && currentAnswer.length === 0)) &&
      !Object.prototype.hasOwnProperty.call(allAnswers, "q_qualities")
    ) {
      const selectedRisks = (allAnswers.q_risks as ImportanceRating[]) || [];
      if (selectedRisks.length > 0) {
        // Get quality IDs that should be preselected based on risks with their max importance
        const preselectedQualities = getQualitiesForRisks(selectedRisks);
        if (preselectedQualities.length > 0) {
          // Convert to ImportanceRating[] format with importance values from related risks
          const qualityRatings = preselectedQualities.map((q) => ({
            id: q.id,
            importance: q.maxImportance || 0,
          }));
          onChange(qualityRatings);
        }
      }
    }
  }, [questionConfig, allAnswers.q_risks, currentAnswer, onChange, allAnswers]);

  if (!questionConfig) return null;

  const getOptionLabel = (option: {
    value: string;
    label?: string;
  }): string => {
    if (option.label) return option.label;
    if (questionConfig.source_data_key) {
      const item = getItemById(questionConfig.source_data_key, option.value);
      return item?.name || option.value;
    }
    return option.value;
  };

  const handleRadioChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  // Rating handlers for importance
  const handleImportanceChange = (id: string, newValue: number | null) => {
    // Ensure newValue is between 1-5
    const importance =
      newValue === null ? 1 : Math.min(Math.max(1, newValue), 5);

    // Handle risks and qualities ratings
    if (
      questionConfig.id === "q_risks" ||
      questionConfig.id === "q_qualities"
    ) {
      // Start with current ratings or empty array
      const ratings: ImportanceRating[] = Array.isArray(currentAnswer)
        ? [...(currentAnswer as ImportanceRating[])]
        : [];

      // Find if this item is already in the ratings
      const existingIndex = ratings.findIndex((item) => item.id === id);

      if (existingIndex >= 0) {
        if (importance === 1) {
          // Remove item if importance is 1
          ratings.splice(existingIndex, 1);
        } else {
          // Update existing rating
          ratings[existingIndex].importance = importance;
        }
      } else if (importance > 1) {
        // Add new rating if it's important
        ratings.push({ id, importance });
      }

      onChange(ratings);
    }
  };

  // Get importance value for an item
  const getImportanceValue = (id: string): number => {
    if (!currentAnswer || !Array.isArray(currentAnswer)) return 1;

    // Find rating for this id
    const rating = (currentAnswer as ImportanceRating[]).find(
      (item) => item.id === id,
    );
    return rating ? rating.importance : 1;
  };

  return (
    <Box>
      <FormControl component="fieldset" variant="standard" fullWidth>
        <FormLabel component="legend" sx={{ mb: 1 }}>
          {questionConfig.text}
        </FormLabel>

        {questionConfig.type === "single-select" && (
          <RadioGroup
            aria-labelledby={`${questionConfig.id}-label`}
            name={questionConfig.id}
            value={currentAnswer || ""}
            onChange={handleRadioChange}
          >
            {questionConfig.options.map((option) => (
              <FormControlLabel
                key={option.value}
                value={option.value}
                control={<Radio />}
                label={getOptionLabel(option)}
              />
            ))}
          </RadioGroup>
        )}

        {questionConfig.type === "multi-select" && (
          <FormGroup>
            {questionConfig.options.map((option) => {
              // For qualities, check if they were pre-selected from risks
              const isQualityFromRisk =
                questionConfig.id === "q_qualities" &&
                Array.isArray(allAnswers.q_risks) &&
                allAnswers.q_risks.length > 0;

              // Get related risks for this quality (if any)
              const qualityWithRisks = isQualityFromRisk
                ? getQualitiesForRisks(
                    allAnswers.q_risks as ImportanceRating[],
                  ).find((q) => q.id === option.value)
                : null;

              const relatedRiskNames =
                qualityWithRisks?.sourceRisks.map((riskId) => {
                  const risk = getItemById("risks", riskId);
                  return risk ? risk.name : riskId;
                }) || [];

              return (
                <Box
                  key={option.value}
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    mb: 2.5,
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <Typography variant="body1" sx={{ mr: 2 }}>
                      {getOptionLabel(option)}
                    </Typography>

                    {relatedRiskNames.length > 0 && (
                      <Chip
                        icon={<StarsIcon fontSize="small" />}
                        label={`Recommended for risk${relatedRiskNames.length > 1 ? "s" : ""}: ${relatedRiskNames.join(", ")}`}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    )}
                  </Box>

                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      mt: 1.5,
                    }}
                  >
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mr: 1, minWidth: 100 }}
                    >
                      Importance:
                    </Typography>
                    <Rating
                      name={`importance-${option.value}`}
                      value={getImportanceValue(option.value)}
                      onChange={(event, newValue) => {
                        handleImportanceChange(option.value, newValue);
                      }}
                      IconContainerComponent={IconContainer}
                      max={5}
                      sx={{ color: "primary.main" }}
                    />
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ ml: 1 }}
                    >
                      {customIcons[getImportanceValue(option.value)]?.label ||
                        "Not important"}
                    </Typography>
                  </Box>
                </Box>
              );
            })}
          </FormGroup>
        )}
      </FormControl>
    </Box>
  );
};
export default QuestionStep;
