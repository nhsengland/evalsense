import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export interface LaTeXProps {
  children: string;
}

export default function LaTeX({ children }: LaTeXProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{ p: "span" }}
    >
      {"$$" + children + "$$"}
    </ReactMarkdown>
  );
}
