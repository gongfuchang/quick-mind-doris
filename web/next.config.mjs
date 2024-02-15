export default (phase, { defaultConfig }) => {
  const env = process.env.NODE_ENV;
  /**
   * @type {import("next").NextConfig}
   */
  if (env === "production") {
    return {
      output: "export",
      assetPrefix: "/ui/",
      basePath: "/ui",
      distDir: "../ui",
    };
  } else {
    return {
      reactStrictMode: false,
      async rewrites() {
        return [
          {
            source: "/query",
            destination: "http://127.0.0.1:8000/query" // Proxy to Backend
          },
        ];
      },
    };
  }
};
