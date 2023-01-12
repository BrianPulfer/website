import React from "react";

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import "./NoMatch.css";
import trackPage from "../../utilities/ga/ga";

class NoMatch extends React.Component {
  render() {
    trackPage();

    return (
      <div className={"noMatch-div"}>
        <BPTitle text={"Hmmm, that page doesn't seem to exist 🤔"} />
        <BPParagraph
          text={"Check the URL, there might be a typo!"}
          size={"26px"}
        />
      </div>
    );
  }
}

export default NoMatch;
