export type CoverageLevel = "Very Good" | "Good" | "Partial" | "Poor";
export type ReferenceRequirement = "required" | "optional" | "not applicable";
export type ItemType = "task" | "quality" | "risk" | "category";

export interface BaseItem {
  id: string;
  name: string;
  description?: string;
  type: ItemType;
}
export type Category = BaseItem;
export type Task = BaseItem;
export type Quality = BaseItem;
export interface Risk extends BaseItem {
  related_qualities?: string[];
}

export interface MethodQualityCoverage {
  id: string;
  coverage: CoverageLevel;
}

export interface MethodRiskCoverage {
  id: string;
  coverage: CoverageLevel;
}

export interface Reference {
  name: string;
  url?: string;
  bib_record?: string;
}

export interface Method extends BaseItem {
  category: string;
  description_short: string;
  description_long_file?: string;
  link_implementation?: string | null;
  link_name?: string | null;
  reference_requirement: ReferenceRequirement;
  supported_tasks: string[];
  assessed_qualities: MethodQualityCoverage[];
  identified_risks: MethodRiskCoverage[];
  output_values?: string;
  advantages?: string[];
  disadvantages?: string[];
  references?: Reference[];
  score?: number;
}

// --- Questionnaire Types ---

export interface QuestionOption {
  value: string;
  label?: string;
}

export interface Question {
  id: string;
  text: string;
  type: "single-select" | "multi-select" | "boolean";
  source_data_key?: "tasks" | "qualities" | "risks" | "categories";
  options: QuestionOption[];
  maps_to: keyof Method | string;
  next: string | null;
}

export interface Questionnaire {
  initial_question: string;
  questions: Record<string, Question>;
}

// --- Guide State Types ---

export interface ImportanceRating {
  id: string;
  importance: number; // 1-5 scale where 1 is "not important" and 5 is "very important"
}

export interface GuideAnswers {
  [questionId: string]: string | ImportanceRating[] | undefined;
}

export interface SuggestionsData {
  filteredMethods: Method[];
  desiredQualities: Quality[];
  desiredRisks: Risk[];
}

export interface GuideState {
  activeStepId: string;
  answers: GuideAnswers;
  selectedMethodIds: string[];
  suggestionsData: SuggestionsData;
}

// --- Preset Types ---

export interface Preset {
  id: string;
  name: string;
  description: string;
  guideState: Partial<GuideState>;
}

// --- Catalogue Types ---
export interface CatalogueFilters {
  searchText?: string;
  tasks?: string[];
  qualities?: string[];
  risks?: string[];
  categories?: string[];
}

// --- Coverage Analysis Types ---

export interface CoverageMap {
  [qualityOrRiskId: string]: CoverageLevel;
}

export interface UncoveredItems {
  qualities: Quality[];
  risks: Risk[];
}

export interface CoverageResult {
  coverage: CoverageMap;
  uncovered: UncoveredItems;
  partiallyCovered: UncoveredItems;
}
