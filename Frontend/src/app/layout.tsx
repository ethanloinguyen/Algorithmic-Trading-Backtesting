import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Stock Covariance Analysis",
  description: "Analyze stock correlations and offset delays",
};

export default function RootLayout({ 
  children 
}: { 
  children: React.ReactNode 
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 antialiased">{children}</body>
    </html>
  );
}