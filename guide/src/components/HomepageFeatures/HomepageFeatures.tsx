import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Interactive Guide",
    Svg: require("@site/static/img/undraw_setup-wizard_wzp9.svg").default,
    description: (
      <>
        Interactive guide helping you select the right evaluation methods for
        your use-case.
      </>
    ),
  },
  {
    title: "Evaluation Method Catalogue",
    Svg: require("@site/static/img/undraw_metrics_02ml.svg").default,
    description: (
      <>
        Extensive catalogue of evaluation methods, including descriptions,
        supported tasks, and more.
      </>
    ),
  },
  {
    title: "Evaluation Presets",
    Svg: require("@site/static/img/undraw_select-option_a16s.svg").default,
    description: (
      <>
        Pre-configured presets for common use-cases, simplifying the selection
        of suitable methods.
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
