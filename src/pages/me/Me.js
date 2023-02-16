import React from "react";
import { Button, Image, Row, Col, ListGroup } from "react-bootstrap";

import Avatar from "./img/avatar.png";
import "./Me.css";

import trackPage from "../../utilities/ga/ga";

class Me extends React.Component {
  render() {
    trackPage();

    let mainImage = (
      <Image id={"me-img"} src={Avatar} className={"mx-auto"} fluid />
    );

    return (
      <div className={"me-div"}>
        <div id={"Image"}>
          <Row>
            <Col className={"text-center"}>{mainImage}</Col>
          </Row>
        </div>

        <div id={"About"}>
          <p className="me-title">About</p>
          <p>
            Hey there, this is <b>Brian</b>! 👋
          </p>
          <p>
            I am a Machine Learning practitioner and enthusiast. I am fascinated
            by the potential of these beautiful and elegant algorithms and their
            potential impact on our world Someday in the future, I'd love to use
            Machine Learning to fight the climate crisis, as I believe is one of
            the biggest priorities and challenges of my generation. Before that,
            I am constantly working on becoming a world expert in the field of
            ML and learn interesting 'hacks' in Computer Science in general.
          </p>
          <p>
            I am currently a Ph.D. student in Machine Learning, with a focus on
            anomaly detection and adversarial attacks, at the University of
            Geneva, Switzerland.
          </p>
          <p>
            This is my personal portfolio, where I publish updates on my career,
            projects, publications and more. Hope you enjoy it!
          </p>
        </div>

        <div id={"News"}>
          <p className="me-title">News 📰</p>

          <ListGroup>
          <ListGroup.Item className={"news"}>
              <p className="news-title">January 2023</p>
              <p>
                📃 Our work{" "}
                <i><a href="https://www.brianpulfer.ch">
                  Model vs System Level Testing of Autonomous Driving Systems: A Replication and Extension Study
                </a></i>{" "}
                has been accepted for publication in {" "}
                <a href="https://www.springer.com/journal/10664">
                  Empirical Software Engineering
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">December 2022</p>
              <p>
                🥉 Our work{" "}
                <i><a href="https://arxiv.org/abs/2212.02456">
                  Solving the Weather4cast Challenge via Visual Transformers for
                  3D Images
                </a></i>{" "}
                got us the third place in the{" "}
                <a href="https://www.iarai.ac.at/weather4cast/">
                  2022 NeurIPS Weather4cast competition workshop
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">August 2022</p>
              <p>
                📃 Our work{" "}
                <i><a href="https://arxiv.org/abs/2209.15625">
                  Anomaly localization for copy detection patterns through print
                  estimations
                </a></i>{" "}
                was accepted for publication in the{" "}
                <a href="https://wifs2022.utt.fr/">
                  IEEE International Workshop on Information Forensics &
                  Security
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">June 2022</p>
              <p>
                📃 Our work{" "}
                <i><a href="https://arxiv.org/abs/2206.11793">
                  Authentication of Copy Detection Patterns under Machine
                  Learning Attacks: A Supervised Approach
                </a></i>{" "}
                was accepted for publication in the{" "}
                <a href="https://2022.ieeeicip.org/">
                  29th IEEE International Conference on Image Processing (ICIP)
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">May 2022</p>
              <p>
                🥇 I won the best presentation award for the 2022 edition of the{" "}
                <a href="https://www.fondazionepremio.ch/premiati/">
                  SwissEngineering Award Foundation
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">December 2021</p>
              <p>
                📃 Our work{" "}
                <i><a href="https://arxiv.org/abs/2112.11255">
                  Mind the Gap! A Study on the Transferability of Virtual vs
                  Physical-world Testing of Autonomous Driving Systems
                </a></i>{" "}
                was accepted for publication in the{" "}
                <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32">
                  IEEE Transactions of Software Engineering (TSE)
                </a>
                .
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <p className="news-title">November 2021</p>
              <p>
                👥 I Joined the{" "}
                <a href="http://sip.unige.ch">
                  Stochastic Information Processing (SIP) group
                </a>{" "}
                of the University of Geneva in quality of Ph.D. Student in
                Machine Learning.
              </p>
            </ListGroup.Item>
          </ListGroup>
        </div>

        <div id={"Download"} className={"last"}>
          <p className="me-title">My CV</p>
          <p className="paragraph text-center">
            Feel free to download my CV from the following button.
          </p>

          <Row>
            <a
              className={"col-12 text-center"}
              href={
                process.env.PUBLIC_URL + "/resources/cv/Brian Pulfer CV.pdf"
              }
              download
            >
              <Button>
                <i className={"fa fa-download"}></i>
                Download my CV
              </Button>
            </a>
          </Row>
        </div>
      </div>
    );
  }
}

export default Me;
