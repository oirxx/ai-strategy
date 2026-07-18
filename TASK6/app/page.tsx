import type { Metadata } from "next";
import Dashboard from "./components/Dashboard";
import data from "./data/dashboard_data.json";

export const metadata: Metadata = {
  title: "Quant Lab 06｜机器学习交易策略",
  description:
    "从财务因子、样本外分类到季度调仓回测：逻辑回归、决策树与随机森林的完整量化研究实验。",
};

export default function Home() {
  return <Dashboard data={data} />;
}
