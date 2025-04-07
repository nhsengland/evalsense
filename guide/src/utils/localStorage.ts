const GUIDE_STATE_KEY = "llmEvalGuideState";
const PRESET_STATE_KEY = "llmEvalPresetToLoad";

export function saveGuideState(state) {
  try {
    const serializedState = JSON.stringify(state);
    localStorage.setItem(GUIDE_STATE_KEY, serializedState);
  } catch (err) {
    console.error("Could not save state to localStorage", err);
  }
}

export function loadGuideState() {
  try {
    const serializedState = localStorage.getItem(GUIDE_STATE_KEY);
    if (serializedState === null) {
      return undefined; // No state saved
    }
    return JSON.parse(serializedState);
  } catch (err) {
    console.error("Could not load state from localStorage", err);
    return undefined;
  }
}

export function clearGuideState() {
  try {
    localStorage.removeItem(GUIDE_STATE_KEY);
  } catch (err) {
    console.error("Could not clear state from localStorage", err);
  }
}

export function setPresetToLoad(presetGuideState) {
  try {
    localStorage.setItem(PRESET_STATE_KEY, JSON.stringify(presetGuideState));
  } catch (err) {
    console.error("Could not save preset state", err);
  }
}

export function loadAndClearPreset() {
  try {
    const presetState = localStorage.getItem(PRESET_STATE_KEY);
    if (presetState) {
      localStorage.removeItem(PRESET_STATE_KEY);
      return JSON.parse(presetState);
    }
    return undefined;
  } catch (err) {
    console.error("Could not load preset state", err);
    localStorage.removeItem(PRESET_STATE_KEY);
    return undefined;
  }
}
