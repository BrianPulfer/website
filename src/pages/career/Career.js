import React from "react";

import "./Career.css";
import { Image, Row, Col } from "react-bootstrap";

import Army from "./img/army.png";
import SUPSI from "./img/supsi.png";
import USI from "./img/usi.png";
import UNIGE from "./img/unige.png";
import trackPage from "../../utilities/ga/ga";

class Career extends React.Component {
  render() {
    trackPage();
    return (
      <div className={"career-div"}>
        <div className={"career-step"}>
          <Row>
            <Col className={"text-center"}>
              <Image src={UNIGE} className={"career-img"} fluid />
            </Col>
          </Row>
          <p className="career-title">
            Ph.D. Student in Machine Learning (2021-)
          </p>
          <p className="paragraph">
            Since November 1st, 2021, I am a Ph.D. student in Machine Learning
            in the Stochastic Information Processing (SIP){" "}
            <a href="http://sip.unige.ch">group</a> at the University of Geneva,
            Switzerland.
          </p>
        </div>

        <div className={"career-step"}>
          <Row>
            <Col className={"text-center"}>
              <Image src={USI} className={"career-img"} fluid />
            </Col>
          </Row>
          <p className="career-title">
            Internship as Machine Learning Engineer (July & August 2020)
          </p>
          <p className="paragraph">
            During the summer of 2020, I have been working as a Machine Learning
            Engineer by the University of Southern Switzerland (USI) on a
            project regarding image classification, object detection and
            segmentation. During this short but enriching experience I have been
            working daily with frameworks such as Pytorch, Tensorflow's Keras
            and OpenCV. With these frameworks, operations such as image
            processing, data augmentation and transfer-learning were carried
            out.
          </p>
        </div>

        <div className={"career-step"}>
          <Row>
            <Col className={"text-center"}>
              <Image src={USI} className={"career-img"} fluid />
            </Col>
          </Row>
          <p className="career-title">
            Master in Artificial Intelligence (2019 - 2021)
          </p>
          <p className="paragraph">
            From September 2019, I have been studying Machine Learning and Deep
            Learning in the Master course offered by the University of Southern
            Switzerland (USI). By learning about
            supervised/unsupervised/reinforcement learning, deep neural networks
            and fascinating mathematical and statistical concepts, I have grown
            a passion for the field. I graduated in June 2021 with a GPA of{" "}
            <b>9.1/10</b>.
          </p>
        </div>

        <div className={"career-step"}>
          <Row>
            <Col className={"text-center"}>
              <Image src={SUPSI} className={"career-img"} fluid />
            </Col>
          </Row>
          <p className="career-title">
            Bachelor in Computer Science (2016 - 2019)
          </p>
          <p className="paragraph">
            From September 2016 to September 2019 I have been studying computer
            science by the University of Applied Sciences of Southern
            Switzerland (SUPSI - Scuola Universitaria Professionale della
            Svizzera Italiana) where I obtained my degree in Computer Science.
            During these 3 years I have learned a lot both in mathematics
            subjects, such as Linear Algebra, Calculus, Probability, Statistics,
            both in computer science subjects such as algorithms and data
            structures, software engineering, design patterns, agile
            development, through many practical projects.
          </p>
        </div>

        <div className={"career-step"}>
          <Row>
            <Col className={"text-center"}>
              <Image src={Army} className={"career-img"} fluid />
            </Col>
          </Row>
          <p className="career-title">
            Service by the Swiss Army (2015 - 2016)
          </p>
          <p className="paragraph">
            From march 2015 to january 2016 I have been serving the Swiss Army
            (10 months) concluding my obligations as a swiss citizen and
            obtaining the grade of soldier. In November 2019, I was finally
            completely discharged of any duty related to the army.
          </p>
        </div>
      </div>
    );
  }
}

export default Career;
