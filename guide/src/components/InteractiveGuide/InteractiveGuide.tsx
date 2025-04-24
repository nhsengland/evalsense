import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Stepper from "@mui/material/Stepper";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import questionnaireData from "@site/src/data/questionnaire.json";
import QuestionStep from "../QuestionStep/QuestionStep";
import SuggestionStep from "../SuggestionStep/SuggestionStep";
import SummaryReport from "@site/src/components/SummaryReport/SummaryReport";
import WelcomeStep from "../WelcomeStep/WelcomeStep";
import { filterAndRankMethods } from "@site/src/utils/evaluationLogic";
import {
  saveGuideState,
  loadGuideState,
  clearGuideState,
  loadAndClearPreset,
} from "@site/src/utils/localStorage";
import {
  Alert,
  IconButton,
  Step,
  StepLabel,
  Tooltip,
  Snackbar,
} from "@mui/material";
import ReplayIcon from "@mui/icons-material/Replay";
import { ImportanceRating } from "@site/src/types/evaluation.types";

const getQuestionConfig = (qId) => questionnaireData.questions[qId];

const steps = [
  { id: "welcome", label: "Welcome" },
  { id: "q_task_type", label: "Task" },
  { id: "q_risks", label: "Risks" },
  { id: "q_qualities", label: "Qualities" },
  { id: "q_references", label: "References" },
  { id: "suggestions", label: "Suggestions" },
  { id: "summary", label: "Summary" },
];

const initialState = {
  activeStepId: questionnaireData.initial_question,
  answers: {},
  selectedMethodIds: [],
  suggestionsData: {
    filteredMethods: [],
    desiredQualities: [],
    desiredRisks: [],
  },
};

export default function InteractiveGuide() {
  const [guideInitialized, setGuideInitialized] = useState(false);
  const [guideState, setGuideState] = useState(initialState);

  const { activeStepId, answers, selectedMethodIds, suggestionsData } =
    guideState;

  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const activeStepIndex = steps.findIndex((step) => step.id === activeStepId);
  const currentQuestionConfig = getQuestionConfig(activeStepId);

  useEffect(() => {
    if (!guideInitialized) {
      const presetState = loadAndClearPreset();
      if (presetState) {
        if (presetState.activeStepId === "suggestions") {
          const results = filterAndRankMethods(presetState.answers);
          presetState.suggestionsData = results;
        }
        return { ...initialState, ...presetState };
      }
      const loadedState = loadGuideState();
      setGuideInitialized(true);
      setGuideState(loadedState || initialState);
    }
  }, [guideInitialized]);

  useEffect(() => {
    saveGuideState({
      activeStepId,
      answers,
      selectedMethodIds,
      suggestionsData,
    });
  }, [activeStepId, answers, selectedMethodIds, suggestionsData]);

  useEffect(() => {
    if (
      guideState.activeStepId === "suggestions" &&
      guideState.suggestionsData.filteredMethods.length === 0 &&
      !isLoadingSuggestions
    ) {
      setIsLoadingSuggestions(true);
      const results = filterAndRankMethods(guideState.answers);
      updateState({ suggestionsData: results });
      setIsLoadingSuggestions(false);
    }
  }, [
    guideState.activeStepId,
    guideState.suggestionsData,
    guideState.answers,
    isLoadingSuggestions,
  ]);

  const updateState = (newState) => {
    setGuideState((prevState) => ({ ...prevState, ...newState }));
  };

  const handleAnswerChange = (
    questionId: string,
    answer: string | string[] | ImportanceRating[],
  ) => {
    updateState({ answers: { ...answers, [questionId]: answer } });
  };

  const handleSelectMethod = (methodId) => {
    updateState({
      selectedMethodIds: selectedMethodIds.includes(methodId)
        ? selectedMethodIds.filter((id) => id !== methodId)
        : [...selectedMethodIds, methodId],
    });
  };

  const setActiveStepId = (newStepId) => {
    updateState({ activeStepId: newStepId });
  };

  const handleReset = () => {
    clearGuideState();
    setGuideState(initialState);
    setShowResetConfirm(true);
  };

  const handleNext = () => {
    let nextStepId = null;
    if (currentQuestionConfig?.next) {
      nextStepId = currentQuestionConfig.next;
    } else if (activeStepId === "suggestions") {
      nextStepId = "summary";
    } else if (activeStepId === "summary") {
      console.log("Final Plan:", { answers, selectedMethodIds });
      return;
    }

    if (nextStepId) {
      if (nextStepId === "suggestions") {
        setIsLoadingSuggestions(true);
        // Need to pass current answers to the logic function
        const results = filterAndRankMethods(answers);
        // Update suggestionsData within the main state object
        updateState({ suggestionsData: results });
        setIsLoadingSuggestions(false);
      }
      setActiveStepId(nextStepId);
    }
  };

  const handleBack = () => {
    if (activeStepIndex > 0) {
      setActiveStepId(steps[activeStepIndex - 1].id);
    }
  };

  const renderStepContent = () => {
    if (isLoadingSuggestions) {
      return (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      );
    }

    switch (activeStepId) {
      case "welcome":
        return <WelcomeStep />;
      case "suggestions":
        return (
          <SuggestionStep
            suggestedMethods={suggestionsData.filteredMethods}
            desiredQualities={suggestionsData.desiredQualities}
            desiredRisks={suggestionsData.desiredRisks}
            selectedMethodIds={selectedMethodIds}
            onSelectMethod={handleSelectMethod}
          />
        );
      case "summary":
        return (
          <SummaryReport
            answers={answers}
            selectedMethodIds={selectedMethodIds}
          />
        );
      default:
        if (currentQuestionConfig) {
          return (
            <QuestionStep
              questionConfig={currentQuestionConfig}
              currentAnswer={answers[activeStepId]}
              onChange={(answer) => handleAnswerChange(activeStepId, answer)}
              allAnswers={answers}
            />
          );
        }
        return <Typography>Configuration Error or Unknown Step</Typography>;
    }
  };

  return (
    <Box sx={{ width: "100%" }}>
      <Snackbar
        open={showResetConfirm}
        autoHideDuration={5000}
        onClose={() => setShowResetConfirm(false)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={() => setShowResetConfirm(false)}
          severity="info"
          variant="filled"
          sx={{ width: "100%" }}
        >
          Progress has been reset.
        </Alert>
      </Snackbar>

      <Stepper
        activeStep={activeStepIndex}
        alternativeLabel
        sx={{ mb: 3, mt: 2 }}
      >
        {steps.map((step) => (
          <Step key={step.id}>
            <StepLabel>{step.label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box sx={{ mt: 2, mb: 1, minHeight: "250px" }}>{renderStepContent()}</Box>

      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          pt: 2,
          alignItems: "center",
        }}
      >
        <Button
          color="inherit"
          disabled={activeStepIndex === 0}
          onClick={handleBack}
          sx={{ mr: 1 }}
        >
          Back
        </Button>

        <Tooltip title="Reset Progress">
          <IconButton onClick={handleReset} color="warning" size="small">
            <ReplayIcon />
          </IconButton>
        </Tooltip>

        <Box sx={{ flex: "1 1 auto" }} />
        <Button
          onClick={handleNext}
          disabled={
            currentQuestionConfig &&
            currentQuestionConfig.type === "single-select" &&
            !answers[activeStepId]
          }
        >
          {activeStepId === "summary" ? "Finish" : "Next"}
        </Button>
      </Box>
    </Box>
  );
}
