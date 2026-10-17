/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        govBlue: {
          50: '#F5FAFF',
          100: '#E3F2FD',
          200: '#BBDEFB',
          500: '#2196F3',
          600: '#1565C0', // Primary Blue
          700: '#10499C',
          800: '#0D47A1', // Dark Blue Header
          900: '#0A337A',
        },
        agriSuccess: '#2E7D32',
        agriWarning: '#ED8B00',
        agriDanger: '#C62828',
      }
    },
  },
  plugins: [],
}
