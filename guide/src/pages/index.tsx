import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures/HomepageFeatures";
import Heading from "@theme/Heading";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/guide"
            style={{ marginRight: "2rem" }}
          >
            📖 Interactive Evaluation Guide
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://nhsengland.github.io/evalsense/docs/"
          >
            🛠️ Library Documentation
          </Link>
        </div>
      </div>
    </header>
  );
}

function ProjectLeadership() {
  return (
    <section className="padding-vert--lg" aria-labelledby="project-leadership">
      <div className="container">
        <Heading as="h2" id="project-leadership" className="text--center">
          Project leadership
        </Heading>
        <p className="text--center margin-bottom--lg">
          EvalSense is owned and managed by NHS England&apos;s Data Science and
          Applied AI team.
        </p>
        <div className="row">
          <div className="col col--6">
            <div className="padding-horiz--md">
              <Heading as="h3">
                <Link href="https://github.com/sylvesterkaczmarek">
                  Sylvester Kaczmarek
                </Link>
              </Heading>
              <p>
                <strong>Scientific and Technical Lead</strong>
              </p>
              <p>
                Leads current scientific and technical development and
                maintenance, including evaluation correctness, reproducibility
                and reliability.
              </p>
            </div>
          </div>
          <div className="col col--6">
            <div className="padding-horiz--md">
              <Heading as="h3">
                <Link href="https://github.com/adamdejl">Adam Dejl</Link>
              </Heading>
              <p>
                <strong>Creator</strong>
              </p>
              <p>
                Created EvalSense through an NHS England PhD Data Science
                internship.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout title="Welcome" description={`${siteConfig.tagline}`}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ProjectLeadership />
      </main>
    </Layout>
  );
}
