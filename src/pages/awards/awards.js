import React from "react";

import "./awards.css";
import trackPage from "../../utilities/ga/ga";

class Awards extends React.Component {
  render() {
    trackPage();

    return (
      <div className={"awards-container"}>
        <div className={"award"}>
          <p className="award-title">🥇 Swiss Engineering Award (2022)</p>
          <p className="paragraph">
            In 2022, I was awarded the 'best presentation award' from the Swiss
            Engineering Ticino foundation for my master thesis on self-driving
            cars.
          </p>
        </div>

        <div className={"award"}>
          <p className="award-title">🥇 SODESKA Scholarship (2021)</p>
          <p className="paragraph">
            In 2021, while studying for my master in Artificial Intelligence, I
            won the SODESKA scholarship awarded to the top 5 swiss students of
            the University of Southern Switzerland in a master course.
          </p>
        </div>
      </div>
    );
  }
}

export default Awards;
