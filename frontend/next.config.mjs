/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The app is fully client-rendered (fetches /meta + /predict in the browser),
  // so we export a static site (-> out/) that any static host can serve. The
  // backend URL is injected at build time via NEXT_PUBLIC_API_BASE_URL.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
