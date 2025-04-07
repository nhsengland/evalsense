import taskDataJson from "@site/src/data/tasks.json";
import qualityDataJson from "@site/src/data/qualities.json";
import riskDataJson from "@site/src/data/risks.json";
import categoryDataJson from "@site/src/data/categories.json";
import methodDataJson from "@site/src/data/methods.json";

import {
  Task,
  Quality,
  Risk,
  Category,
  Method,
  BaseItem,
} from "../types/evaluation.types";

const taskData: Task[] = taskDataJson as Task[];
const qualityData: Quality[] = qualityDataJson as Quality[];
const riskData: Risk[] = riskDataJson as Risk[];
const categoryData: Category[] = categoryDataJson as Category[];
const methodData: Method[] = methodDataJson as Method[];

type DataKey = "tasks" | "qualities" | "risks" | "categories" | "methods";

const dataMap: Record<DataKey, BaseItem[]> = {
  tasks: taskData,
  qualities: qualityData,
  risks: riskData,
  categories: categoryData,
  methods: methodData,
};

const dataLookups: Record<DataKey, Map<string, BaseItem>> = {
  tasks: new Map(taskData.map((item) => [item.id, item])),
  qualities: new Map(qualityData.map((item) => [item.id, item])),
  risks: new Map(riskData.map((item) => [item.id, item])),
  categories: new Map(categoryData.map((item) => [item.id, item])),
  methods: new Map(methodData.map((item) => [item.id, item])),
};

export function getData(key: DataKey): BaseItem[] {
  return dataMap[key] || [];
}

export function getItemById<T extends BaseItem>(
  key: DataKey,
  id: string
): T | undefined {
  return dataLookups[key]?.get(id) as T | undefined;
}

export function getItems<T extends BaseItem>(key: DataKey): T[] {
  const data = dataMap[key];
  return data.map((item) => item as T);
}

export function getItemsByIds<T extends BaseItem>(
  key: DataKey,
  ids: string[] | undefined
): T[] {
  if (!Array.isArray(ids)) return [];
  const lookup = dataLookups[key];
  if (!lookup) return [];
  return ids.map((id) => lookup.get(id)).filter(Boolean) as T[];
}
