import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "EvalSense",
  tagline:
    "Comprehensive guidance and tooling for evaluating large language models (LLMs)",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://nhsengland.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/evalsense/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "nhsengland", // Usually your GitHub org/user name.
  projectName: "evalsense", // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: false,
        blog: false,
        theme: {
          customCss: "src/css/custom.css",
        },
        pages: {
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
  ],

  themeConfig: {
    navbar: {
      title: "EvalSense",
      logo: {
        alt: "EvalSense Logo",
        src: "img/logo.svg",
      },
      items: [
        { to: "/guide", label: "Interactive Guide", position: "left" },
        { to: "/catalogue", label: "Method Catalogue", position: "left" },
        { to: "/presets", label: "Evaluation Presets", position: "left" },
        {
          href: "https://nhsengland.github.io/evalsense/docs/",
          label: "Library Documentation",
          position: "left",
        },
        {
          href: "https://github.com/nhsengland/evalsense",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Evaluation Resources",
          items: [
            {
              label: "Interactive Guide",
              to: "/guide",
            },
            {
              label: "Method Catalogue",
              to: "/catalogue",
            },
            {
              label: "Evaluation Presets",
              to: "/presets",
            },
          ],
        },
        {
          title: "EvalSense Library",
          items: [
            {
              label: "Documentation",
              to: "https://nhsengland.github.io/evalsense/docs/",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/nhsengland/evalsense",
            },
            {
              label: "NHS England Data Science",
              href: "https://nhsengland.github.io/datascience/",
            },
          ],
        },
      ],
      copyright: `<a href="https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/" style="color:#5fb1ff">© Crown copyright</a> ${new Date().getFullYear()} NHS England. Available under the terms of the <a href="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/" style="color:#5fb1ff">Open Government Licence v3.0</a>.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    colorMode: {
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
