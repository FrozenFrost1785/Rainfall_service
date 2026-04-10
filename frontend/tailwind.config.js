export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        deep:    '#050D18',
        ocean:   '#071525',
        slate2:  '#0A1E30',
        border2: '#0F2A3D',
        teal:    '#00C9A7',
        aqua:    '#00E5FF',
        emerald: '#00B37E',
        storm:   '#FF6B35',
        warn:    '#FFD166',
        mist:    '#7FDBCA',
      },
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        mono:    ['"Fira Code"', '"Fira Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
