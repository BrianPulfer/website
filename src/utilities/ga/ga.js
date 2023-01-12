import ReactGA from "react-ga";

const G_TOKEN = "G-BH82F18037";

function trackPage() {
  let url = window.location.pathname + window.location.hash;
  ReactGA.initialize(G_TOKEN);
  ReactGA.set({ url });
  ReactGA.pageview(url);
}

export default trackPage;
