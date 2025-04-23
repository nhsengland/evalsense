import React, { useEffect } from "react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Checkbox from "@mui/material/Checkbox";
import FormGroup from "@mui/material/FormGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import StarsIcon from "@mui/icons-material/Stars";
import { getItemById } from "@site/src/utils/dataLoaders";
import { getQualitiesForRisks } from "@site/src/utils/evaluationLogic";
import { Question, GuideAnswers } from "@site/src/types/evaluation.types";

interface QuestionStepProps {
  questionConfig: Question;
  currentAnswer: GuideAnswers[string];
  onChange: (answer: string | string[]) => void;
  allAnswers?: GuideAnswers; // To access previous answers like risks
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
      const selectedRisks = (allAnswers.q_risks as string[]) || [];
      if (selectedRisks.length > 0) {
        // Get quality IDs that should be preselected based on risks
        const preselectedQualities = getQualitiesForRisks(selectedRisks);
        if (preselectedQualities.length > 0) {
          onChange(preselectedQualities.map((q) => q.id));
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

  const handleRadioChange = (event) => {
    onChange(event.target.value);
  };

  const handleCheckboxChange = (event) => {
    const { value, checked } = event.target;
    const currentValues = Array.isArray(currentAnswer) ? currentAnswer : [];
    if (checked) {
      onChange([...currentValues, value]);
    } else {
      onChange(currentValues.filter((v) => v !== value));
    }
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
                ? getQualitiesForRisks(allAnswers.q_risks as string[]).find(
                    (q) => q.id === option.value,
                  )
                : null;

              const relatedRiskNames =
                qualityWithRisks?.sourceRisks.map((riskId) => {
                  const risk = getItemById("risks", riskId);
                  return risk ? risk.name : riskId;
                }) || [];

              return (
                <Box
                  key={option.value}
                  sx={{ display: "flex", alignItems: "center", mb: 0.5 }}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={
                          Array.isArray(currentAnswer) &&
                          currentAnswer.includes(option.value)
                        }
                        onChange={handleCheckboxChange}
                        value={option.value}
                      />
                    }
                    label={getOptionLabel(option)}
                  />

                  {relatedRiskNames.length > 0 && (
                    <Chip
                      icon={<StarsIcon fontSize="small" />}
                      label={`Recommended for risk${relatedRiskNames.length > 1 ? "s" : ""}: ${relatedRiskNames.join(", ")}`}
                      size="small"
                      color="primary"
                      variant="outlined"
                      sx={{ ml: 1 }}
                    />
                  )}
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
