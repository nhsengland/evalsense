import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";
import InteractiveGuideSvg from "@site/static/img/undraw_setup-wizard_wzp9.svg";
import EvaluationMethodSvg from "@site/static/img/undraw_metrics_02ml.svg";
import LibrarySvg from "@site/static/img/undraw_developer-activity_4zqd.svg";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "LLM Evaluation Library",
    Svg: LibrarySvg,
    description: (
      <>
        Python library for systematic evaluation of large language models on
        open-ended generation tasks.
      </>
    ),
  },
  {
    title: "Interactive Guide",
    Svg: InteractiveGuideSvg,
    description: (
      <>
        Interactive guide helping you select the right evaluation methods for
        your use-case.
      </>
    ),
  },
  {
    title: "Evaluation Method Catalogue",
    Svg: EvaluationMethodSvg,
    description: (
      <>
        Extensive catalogue of evaluation methods, including descriptions,
        supported tasks, and more.
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
