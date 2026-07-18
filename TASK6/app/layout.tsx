import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Noto_Sans_SC } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-sans-latin", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });
const noto = Noto_Sans_SC({
  variable: "--font-sans-cn",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
});

export const metadata: Metadata = {
  title: "Quant Lab 06｜机器学习交易策略",
  description:
    "一份可交互的机器学习选股研究：21个因子、3类模型、3期样本外季度回测。",
  openGraph: {
    title: "Quant Lab 06｜机器学习交易策略",
    description: "预测只是开始，收益才是验收。",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Quant Lab 06｜机器学习交易策略",
    description: "从概率到仓位，从模型到回测。",
    images: ["/og.png"],
  },
};

export const viewport: Viewport = {
  themeColor: "#07110f",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body className={`${geist.variable} ${mono.variable} ${noto.variable}`}>
        {children}
      </body>
    </html>
  );
}
