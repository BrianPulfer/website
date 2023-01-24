import React from "react";
import { Button, Image, Row, Col, ListGroup } from "react-bootstrap";

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import Avatar from "./img/avatar.png";
import "./Me.css";

import trackPage from "../../utilities/ga/ga";

class Me extends React.Component {
  render() {
    trackPage();

    const titleSize = "500%";
    const paragraphSize = "180%";

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
          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"About"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={"Hey there, this is Brian! 👋"}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "I am a Machine Learning practitioner and enthusiast. I am fascinated by the potential of these beautiful and elegant algorithms and their potential impact on our world. " +
              "Someday in the future, I'd love to use Machine Learning to fight the climate crisis, as I believe is one of the biggest priorities and challenges of my generation. " +
              "Before that, I am constantly working on becoming a world expert in the field of ML and learn interesting 'hacks' in Computer Science in general."
            }
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "I am currently a Ph.D. student in Machine Learning, with a focus on anomaly detection and adversarial attacks, by the University of Geneva, Switzerland."
            }
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "This is my personal portfolio, where I publish updates on my career, projects, publications and more. I hope you enjoy it!"
            }
          />

        </div>

        <div id={"News"}>
          <BPTitle
              classes={"h1 bp-title-me"}
              size={titleSize}
              color={"black"}
              text={"News 📰"}
              underline={true}
            />

          <ListGroup>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"December 2022"} size={paragraphSize} side={true}/>
              <p>
               🥉 Our work <a href="https://arxiv.org/abs/2212.02456">Solving the Weather4cast Challenge via Visual Transformers for 3D Images</a> got us the third place in the <a href="https://www.iarai.ac.at/weather4cast/">2022 NeurIPS Weather4cast competition workshop</a>.
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"August 2022"} size={paragraphSize} side={true}/>
              <p>
               📃 Our work <a href="https://arxiv.org/abs/2209.15625">Anomaly localization for copy detection patterns through print estimations</a> was accepted for publication in the <a href="https://wifs2022.utt.fr/">IEEE International Workshop on Information Forensics & Security</a>.
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"June 2022"} size={paragraphSize} side={true}/>
              <p>
               📃 Our work <a href="https://arxiv.org/abs/2206.11793">Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach</a> was accepted for publication in the <a href="https://2022.ieeeicip.org/">29th IEEE International Conference on Image Processing (ICIP)</a>.
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"May 2022"} size={paragraphSize} side={true}/>
              <p>
               🥇 I won the best presentation award for the 2022 edition of the <a href="https://www.fondazionepremio.ch/premiati/">SwissEngineering Award Foundation</a>.
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"December 2021"} size={paragraphSize} side={true}/>
              <p>
               📃 Our work <a href="https://arxiv.org/abs/2112.11255">Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems</a> was accepted for publication in the <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32">IEEE Transactions of Software Engineering (TSE)</a>.
              </p>
            </ListGroup.Item>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"November 2021"} size={paragraphSize} side={true}/>
              <p>
               👥 I Joined the <a href="http://sip.unige.ch">Stochastic Information Processing (SIP) group</a> of the University of Geneva in quality of Ph.D. Student in Machine Learning.
              </p>
            </ListGroup.Item>
          </ListGroup>
        </div>

        <div id={"Download"} className={"last"}>
          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"My CV"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "You can download my CV from the following button."
            }
          />

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
