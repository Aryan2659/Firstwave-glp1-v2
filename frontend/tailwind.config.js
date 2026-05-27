/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Fraunces"', "Georgia", "serif"],
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: {
          50: "#f7f5f0",
          100: "#ebe7dc",
          200: "#d6cfbb",
          300: "#b8ad8e",
          400: "#9b8d65",
          500: "#7a6d4e",
          600: "#5d533c",
          700: "#3f3829",
          800: "#28231a",
          900: "#1a1611",
        },
        accent: {
          50: "#fff7ec",
          100: "#ffe9c8",
          200: "#ffd28a",
          300: "#ffb44b",
          400: "#ff961f",
          500: "#f97306",
          600: "#dd4f00",
          700: "#b73706",
          800: "#922c0c",
          900: "#76250d",
        },
      },
    },
  },
  plugins: [],
};
