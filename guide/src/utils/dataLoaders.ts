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

type DataLookup = {
  tasks: Map<string, Task>;
  qualities: Map<string, Quality>;
  risks: Map<string, Risk>;
  categories: Map<string, Category>;
  methods: Map<string, Method>;
};

type DataMap = {
  tasks: Task[];
  qualities: Quality[];
  risks: Risk[];
  categories: Category[];
  methods: Method[];
};

type MapValue<T> = T extends Map<string, infer V> ? V : never;

const dataMap: DataMap = {
  tasks: taskData,
  qualities: qualityData,
  risks: riskData,
  categories: categoryData,
  methods: methodData,
};

const dataLookups: DataLookup = {
  tasks: new Map(taskData.map((item) => [item.id, item])),
  qualities: new Map(qualityData.map((item) => [item.id, item])),
  risks: new Map(riskData.map((item) => [item.id, item])),
  categories: new Map(categoryData.map((item) => [item.id, item])),
  methods: new Map(methodData.map((item) => [item.id, item])),
};

export function getData<K extends keyof DataMap>(key: K): DataMap[K] {
  return dataMap[key];
}

export function getItemById<K extends keyof DataLookup>(
  key: K,
  id: string
): MapValue<DataLookup[K]> | undefined {
  const map = dataLookups[key] as Map<string, MapValue<DataLookup[K]>>;
  return map.get(id);
}

export function getItemsByIds<K extends keyof DataLookup>(
  key: K,
  ids: string[]
): MapValue<DataLookup[K]>[] {
  const lookup = dataLookups[key] as Map<string, MapValue<DataLookup[K]>>;
  if (!lookup) return [];
  return ids.map((id) => lookup.get(id)).filter(Boolean);
}
