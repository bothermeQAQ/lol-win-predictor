import "./globals.css";

export const metadata = {
  title: "Draft Lab — Blue-Side Win Model",
  description:
    "Broadcast-grade League of Legends blue-side win predictor. Dead Draft vs Live Map — champion swaps barely move the model, early objectives are an earthquake.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#05060A",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
