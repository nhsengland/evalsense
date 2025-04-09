import { getItemsByIds, getData } from "./dataLoaders";
import {
  Method,
  GuideAnswers,
  SuggestionsData,
  Quality,
  Risk,
  CoverageLevel,
  CoverageResult,
  CoverageMap,
  UncoveredItems,
} from "../types/evaluation.types";

const coverageScore: Record<CoverageLevel, number> = {
  "Very Good": 4,
  Good: 3,
  Partial: 2,
  Poor: 1,
};

export function filterAndRankMethods(answers: GuideAnswers): SuggestionsData {
  const allMethods = getData("methods") as Method[];
  const desiredQualityIds = (answers.q_qualities as string[] | undefined) || [];
  const desiredRiskIds = (answers.q_risks as string[] | undefined) || [];
  const taskType = answers.q_task_type as string | undefined;
  const noReference = answers.q_references === "no";

  const desiredQualities = getItemsByIds("qualities", desiredQualityIds);
  const desiredRisks = getItemsByIds("risks", desiredRiskIds);

  const filtered = allMethods.filter((method) => {
    // 1. Task suitability check
    if (taskType && !method.supported_tasks.includes(taskType)) {
      return false;
    }

    // 2. Reference requirement check
    if (noReference && method.reference_requirement === "required") {
      return false;
    }

    return true;
  });

  // Calculate score for each method based on user's desired qualities/risks
  const scoredMethods = filtered.map((method) => {
    let score = 0;
    desiredQualityIds.forEach((qId) => {
      const coverage = method.assessed_qualities.find(
        (q) => q.id === qId
      )?.coverage;
      score += coverageScore[coverage] || 0;
    });
    desiredRiskIds.forEach((rId) => {
      const coverage = method.identified_risks.find(
        (r) => r.id === rId
      )?.coverage;
      score += coverageScore[coverage] || 0;
    });
    return { ...method, score };
  });

  scoredMethods.sort((a, b) => b.score - a.score);

  return {
    filteredMethods: scoredMethods,
    desiredQualities,
    desiredRisks,
  };
}

/**
 * Calculates coverage of desired qualities/risks by selected methods.
 * @param {string[]} selectedMethodIds - Array of IDs of methods chosen by the user.
 * @param {object[]} desiredQualities - Array of desired quality objects {id, name}.
 * @param {object[]} desiredRisks - Array of desired risk objects {id, name}.
 * @returns {object} - { coverage: { quality/risk_id: best_coverage_level }, uncovered: { qualities: [...], risks: [...] } }
 */
export function calculateCoverage(
  selectedMethodIds: string[],
  desiredQualities: Quality[],
  desiredRisks: Risk[]
): CoverageResult {
  const selectedMethods = getItemsByIds("methods", selectedMethodIds);
  const coverageMap: CoverageMap = {};
  const uncovered: UncoveredItems = {
    qualities: [...desiredQualities],
    risks: [...desiredRisks],
  };

  desiredQualities.forEach((quality) => {
    let bestCoverage = 0; // Use numeric score for comparison
    let bestCoverageLevel = null;
    selectedMethods.forEach((method) => {
      const qualCoverage = method.assessed_qualities.find(
        (q) => q.id === quality.id
      )?.coverage;
      const score = coverageScore[qualCoverage] || 0;
      if (score > bestCoverage) {
        bestCoverage = score;
        bestCoverageLevel = qualCoverage;
      }
    });
    if (bestCoverageLevel) {
      coverageMap[quality.id] = bestCoverageLevel;
      uncovered.qualities = uncovered.qualities.filter(
        (q) => q.id !== quality.id
      );
    }
  });

  desiredRisks.forEach((risk) => {
    let bestCoverage = 0;
    let bestCoverageLevel = null;
    selectedMethods.forEach((method) => {
      const riskCoverage = method.identified_risks.find(
        (r) => r.id === risk.id
      )?.coverage;
      const score = coverageScore[riskCoverage] || 0;
      if (score > bestCoverage) {
        bestCoverage = score;
        bestCoverageLevel = riskCoverage;
      }
    });
    if (bestCoverageLevel) {
      coverageMap[risk.id] = bestCoverageLevel;
      uncovered.risks = uncovered.risks.filter((r) => r.id !== risk.id);
    }
  });

  return { coverage: coverageMap, uncovered };
}
