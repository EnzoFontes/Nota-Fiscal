/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        accent: '#E42313',
        success: '#22C55E',
        border: '#E8E8E8',
        muted: '#7A7A7A',
        placeholder: '#B0B0B0',
        surface: '#FAFAFA',
      },
    },
  },
  plugins: [],
}
