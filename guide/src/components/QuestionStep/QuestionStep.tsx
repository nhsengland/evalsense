import React from "react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Checkbox from "@mui/material/Checkbox";
import FormGroup from "@mui/material/FormGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import Box from "@mui/material/Box";
import { getItemById } from "@site/src/utils/dataLoaders";
import { Question, GuideAnswers } from "@site/src/types/evaluation.types";

interface QuestionStepProps {
  questionConfig: Question;
  currentAnswer: GuideAnswers[string];
  onChange: (answer: string | string[]) => void;
}

const QuestionStep: React.FC<QuestionStepProps> = ({
  questionConfig,
  currentAnswer,
  onChange,
}) => {
  if (!questionConfig) return null;

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
            {questionConfig.options.map((option) => (
              <FormControlLabel
                key={option.value}
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
            ))}
          </FormGroup>
        )}
      </FormControl>
    </Box>
  );
};
export default QuestionStep;
