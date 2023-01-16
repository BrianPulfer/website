// Reference: https://blog.saeloun.com/2022/02/17/how-to-integrate-react-app-with-google-analytics.html

import ReactGA from "react-ga";

const G_TOKEN = "G-BH82F18037";

function trackPage() {
  let path = window.location.hash
  let title = path.substring(2, path.length)
  if (title.length === 0){
    document.title = "Brian Pulfer"
  } else{
    document.title = "Brian Pulfer - " + title;
  }
  ReactGA.initialize(G_TOKEN);
  ReactGA.pageview(path, [], title);
}

export default trackPage;
