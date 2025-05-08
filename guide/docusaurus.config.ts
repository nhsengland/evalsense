import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "LLM Evaluation Guide",
  tagline: "Comprehensive guidance for evaluating Large Language Models",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://your-docusaurus-site.example.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "facebook", // Usually your GitHub org/user name.
  projectName: "docusaurus", // Usually your repo name.

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
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/nhsengland/evalsense/tree/main/guide",
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
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
      title: "LLM Evaluation Guide",
      logo: {
        alt: "LLM Evaluation Guide Logo",
        src: "img/logo.svg",
      },
      items: [
        { to: "/guide", label: "Interactive Guide", position: "left" },
        { to: "/catalogue", label: "Method Catalogue", position: "left" },
        { to: "/presets", label: "Evaluation Presets", position: "left" },
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
          title: "Documentation",
          items: [
            {
              label: "Introduction",
              to: "/docs/intro",
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
