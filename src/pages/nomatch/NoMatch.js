import React from "react";

import "./NoMatch.css";
import trackPage from "../../utilities/ga/ga";

class NoMatch extends React.Component {
  render() {
    trackPage();

    return (
      <div className={"noMatch-div"}>
        <p className="nomatch-title">
          Hmmm, that page doesn't seem to exist 🤔
        </p>
        <p className="nomatch-paragraph">
          Check the URL, there might be a typo!
        </p>
      </div>
    );
  }
}

export default NoMatch;
