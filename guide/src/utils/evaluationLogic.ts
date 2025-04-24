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
  ImportanceRating,
} from "../types/evaluation.types";

const coverageScore: Record<CoverageLevel, number> = {
  "Very Good": 4,
  Good: 3,
  Partial: 2,
  Poor: 1,
};

export function getQualitiesForRisks(
  riskImportance: ImportanceRating[],
): { id: string; sourceRisks: string[]; maxImportance: number }[] {
  // First gather all the quality IDs that need to be pre-selected
  const qualityMap = new Map<
    string,
    { sourceRisks: string[]; maxImportance: number }
  >();

  // Get risk objects from IDs with their importance
  const riskIds = riskImportance
    .filter((r) => r.importance > 1)
    .map((r) => r.id);
  const risks = getItemsByIds("risks", riskIds) as Risk[];

  // Process each risk's related qualities
  risks.forEach((risk) => {
    const relatedQualityIds = risk.related_qualities || [];
    // Get the importance rating for this risk
    const importance =
      riskImportance.find((r) => r.id === risk.id)?.importance || 1;

    relatedQualityIds.forEach((qualityId) => {
      // Add or update the sourceRisks array and track the max importance
      if (!qualityMap.has(qualityId)) {
        qualityMap.set(qualityId, {
          sourceRisks: [risk.id],
          maxImportance: importance,
        });
      } else {
        const current = qualityMap.get(qualityId);
        if (!current.sourceRisks.includes(risk.id)) {
          current.sourceRisks.push(risk.id);
        }
        // Update max importance if this risk has higher importance
        if (importance > current.maxImportance) {
          current.maxImportance = importance;
        }
      }
    });
  });

  // Convert the map to the result array
  const result = Array.from(qualityMap.entries()).map(
    ([id, { sourceRisks, maxImportance }]) => ({
      id,
      sourceRisks,
      maxImportance,
    }),
  );

  return result;
}

export function filterAndRankMethods(answers: GuideAnswers): SuggestionsData {
  const allMethods = getData("methods") as Method[];

  // Get quality and risk ratings
  const qualityRatings =
    (answers.q_qualities as ImportanceRating[] | undefined) || [];
  const riskRatings = (answers.q_risks as ImportanceRating[] | undefined) || [];

  // Get IDs for qualities and risks that have importance > 1
  const desiredQualityIds = qualityRatings
    .filter((q) => q.importance > 1)
    .map((q) => q.id);
  const desiredRiskIds = riskRatings
    .filter((r) => r.importance > 1)
    .map((r) => r.id);

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

    // Weight scores by importance
    qualityRatings.forEach((qRating) => {
      if (qRating.importance > 1) {
        const coverage = method.assessed_qualities.find(
          (q) => q.id === qRating.id,
        )?.coverage;
        const coverageValue = coverageScore[coverage] || 0;
        score += coverageValue * qRating.importance;
      }
    });

    // Similar for risks
    riskRatings.forEach((rRating) => {
      if (rRating.importance > 1) {
        const coverage = method.identified_risks.find(
          (r) => r.id === rRating.id,
        )?.coverage;
        const coverageValue = coverageScore[coverage] || 0;
        // Weight by importance (0-4 scale)
        score += coverageValue * rRating.importance;
      }
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
 * @param {Quality[]} desiredQualities - Array of desired quality objects.
 * @param {Risk[]} desiredRisks - Array of desired risk objects.
 * @param {ImportanceRating[]} qualityRatings - Array of quality importance ratings.
 * @param {ImportanceRating[]} riskRatings - Array of risk importance ratings.
 * @returns {object} - { coverage: { quality/risk_id: best_coverage_level }, uncovered: { qualities: [...], risks: [...] } }
 */
export function calculateCoverage(
  selectedMethodIds: string[],
  desiredQualities: Quality[],
  desiredRisks: Risk[],
  qualityRatings: ImportanceRating[] = [],
  riskRatings: ImportanceRating[] = [],
): CoverageResult {
  const selectedMethods = getItemsByIds("methods", selectedMethodIds);
  const coverageMap: CoverageMap = {};

  // Filter to only include qualities and risks with importance > 1
  const significantQualities = desiredQualities.filter((q) => {
    const rating = qualityRatings.find((r) => r.id === q.id);
    return rating && rating.importance > 1;
  });

  const significantRisks = desiredRisks.filter((r) => {
    const rating = riskRatings.find((rat) => rat.id === r.id);
    return rating && rating.importance > 1;
  });

  const uncovered: UncoveredItems = {
    qualities: [...significantQualities],
    risks: [...significantRisks],
  };

  significantQualities.forEach((quality) => {
    let bestCoverage = 0; // Use numeric score for comparison
    let bestCoverageLevel = null;
    selectedMethods.forEach((method) => {
      const qualCoverage = method.assessed_qualities.find(
        (q) => q.id === quality.id,
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
        (q) => q.id !== quality.id,
      );
    }
  });

  significantRisks.forEach((risk) => {
    let bestCoverage = 0;
    let bestCoverageLevel = null;
    selectedMethods.forEach((method) => {
      const riskCoverage = method.identified_risks.find(
        (r) => r.id === risk.id,
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
