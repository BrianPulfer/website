import React from "react";
import "./../../../node_modules/video-react/dist/video-react.css";

import { Image, Row, Col } from "react-bootstrap";

import "./Projects.css";

// Images
import Bachelor from "./img/Machine Learning for disambiguation of scientific article authors.png";
import StyleGAN2 from "./img/StyleGAN2 Distillation.png";
import SDR from "./img/Self Driving Robot.gif";
import DLL from "./img/Deep Learning Lab.png";
import ML from "./img/Machine Learning.gif";
import SmartBin from "./img/SmartBin.png";

import NannySearch from "./img/NannySearch.png";
import Tiforma from "./img/Tiforma.png";
import trackPage from "../../utilities/ga/ga";

class Projects extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      videosWidth: window.innerWidth * 0.7,
      videosHeight: window.innerWidth * 0.7 * 0.5625,
    };

    window.onresize = (ev) => {
      this.setState({
        videosWidth: ev.target.innerWidth * 0.7,
        videosHeight: ev.target.innerWidth * 0.7 * 0.5625,
      });
    };
  }

  render() {
    trackPage();

    return (
      <div className={"projects-div"}>
        <div id={"Projects"}>
          <div id={"AI1"} className={"project"}>
            <div>
              <p className="project-title">
                From Simulated to Real Test Environments for Self Driving Cars
              </p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <iframe
                  width={this.state.videosWidth + "px"}
                  height={this.state.videosHeight + "px"}
                  src="https://www.youtube.com/embed/7q2hwzWo7Cw"
                  title="From Simulated to Real Test Environments for Self Driving Cars"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                ></iframe>
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                In my master thesis in Artificial Intelligence, I studied
                testing in the field of self-driving cars through a small-scale
                car and simulator.
              </p>
              <p>
                Through the use of CycleGAN, I propose a method to estimate the
                Cross-Track Error in the real world (important testing metric
                already in use for simulators) and use it to assess whether
                offline and online testing for self-driving cars yields similar
                results, both in a real and simulated environment.
              </p>
              <p>
                Given the enthusiasm that me and my co-supervisor had towards
                this small-scale car, we even organized the first{" "}
                <a href={"https://formulausi.si.usi.ch/2021/"}>
                  <b>FormulaUSI</b>
                </a>{" "}
                event! The goal of the event was to educate participants on
                Artificial Intelligence while racing self-driving small-scale
                cars. We had much fun organizing the event, and I have
                personally grown by such an experience.
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  My master thesis can be downloaded at this{" "}
                  <a
                    href={
                      process.env.PUBLIC_URL +
                      "/resources/docs/Brian Pulfer - From Simulated to Real Test Environments for Self Driving Cars.pdf"
                    }
                  >
                    link
                  </a>
                  .<br />
                  Here's the links to the FormulaUSI competition{" "}
                  <a href={"https://formulausi.si.usi.ch/2021/"}>
                    website
                  </a> and{" "}
                  <a
                    href={
                      "https://www.youtube.com/watch?v=PDeCb4vBEC4&ab_channel=SoftwareInstitute"
                    }
                  >
                    highlights
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI2"} className={"project"}>
            <div>
              <p className="project-title">
                Machine Learning for disambiguation of scientific article
                authors
              </p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={Bachelor} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                This project is an open-source implementation of a classifier
                which goal is to predict whether a pair of scientific articles
                (biomedical articles from the PubMed dataset) belongs to the
                same author or not."
              </p>
              <p>
                The final classifier (Random Forest) used 15 features and had an
                accuracy of 87% with a 10-fold cross-validation. Further studies
                on the datasets revealed that for some combinations of last
                names and initial of first names (namespaces), over 100'000
                articles could be found. This study explains the need for a
                classifier able to distinguish between these authors.
              </p>
            </div>
            <div className={"project-links"}>
              <p>
                The project was my bachelor thesis job commissioned by{" "}
                <b>Hoffmann-La Roche A.G</b>
              </p>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a
                    href={
                      "https://github.com/BrianPulfer/AuthorNameDisambiguation"
                    }
                  >
                    link
                  </a>
                  .<br />
                  You can also visit the study on the Pubmed dataset at the
                  following{" "}
                  <a href={"https://github.com/BrianPulfer/PubMed-Namespacer"}>
                    link
                  </a>
                  .<br />
                  Documentation (Italian Only) of the bachelor's thesis can be
                  downloaded at this{" "}
                  <a
                    href={
                      process.env.PUBLIC_URL +
                      "/resources/docs/Brian Pulfer - Machine Learning for disambiguation of scientific article authors.pdf"
                    }
                  >
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI3"} className={"project"}>
            <div>
              <p className="project-title">StyleGAN2 distillation</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={StyleGAN2} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                Together with my brother, we re-implemented the work of{" "}
                <a href={"https://arxiv.org/abs/2003.03581"}>Evgeny et. al.</a>
                in their 'StyleGAN2 distillation' paper released in 2020. In
                their work, the authors show that a Pix2PixHD network can be
                trained on a synthetic dataset generated by a GAN (StyleGAN2).
              </p>
              <p>
                In the re-implementation provided, we focus on the style-mixing
                task, and manage to train a vanilla Pix2PixHD network into
                creating a style-mixed face starting from 2 pictures of real
                persons.
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  The latest checkpoints of the network can be found{" "}
                  <a
                    href={
                      "https://drive.google.com/drive/folders/1o2Mq7vok4FUB8rOCtHnsT78igfN9GK1S?usp=sharing"
                    }
                  >
                    here
                  </a>
                  . <br />
                  The notebook containing the re-implementation of the project
                  is publicly available at this{" "}
                  <a
                    href={
                      "https://colab.research.google.com/drive/1hxZvml_rbjF62W-9bW39Dap1zJgR9K5w?usp=sharing"
                    }
                  >
                    link
                  </a>
                  .<br />
                  Documentation of how the project was reimplemented can be
                  found at the following{" "}
                  <a
                    href={
                      process.env.PUBLIC_URL +
                      "/resources/docs/Brian Pulfer - StyleGAN2 Distillation Reimplementation.pdf"
                    }
                  >
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI4"} className={"project"}>
            <div>
              <p className="project-title">Self-driving Robot</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"mx-auto prjimg"} src={SDR} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                This project represents the final assignment of the course of
                Robotics held by the University of Southern Switzerland (USI).
                The goal of the project was to use both the theoretical and
                practical knowledge in the field of robotics to implement some
                complex program that would make the robot accomplish some goals.
              </p>
              <p>
                The project me and my team (2 other persons) came up with is a
                self-driving robot that, with the use of CNN, learns to avoid
                obstacles and walls in a virtual environment (ROS + Gazebo).
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a
                    href={
                      "https://github.com/BrianPulfer/Learning-to-drive-by-crashing"
                    }
                  >
                    link
                  </a>
                  .<br />A very brief paper of what has been done during the
                  project can be downloaded at the following{" "}
                  <a
                    href={
                      process.env.PUBLIC_URL +
                      "/resources/docs/Brian Pulfer - Self driving Robot - Final Paper.pdf"
                    }
                  >
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI5"} className={"project"}>
            <div className>
              <p className="project-title">Deep Learning Lab</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={DLL} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                This project is the collection of all activities that were done
                during a University course. With the use of Tensorflow 1, you
                will find various implementations of linear regression, feed
                forward neural networks, recurrent neural networks,
                convolutional neural networks, long short-term memory networks
                and a deep Q-learning algorithm (break-out agent).
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a href={"https://github.com/BrianPulfer/Deep-Learning-Lab"}>
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI6"} className={"project"}>
            <div>
              <p className="project-title">Machine Learning</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={ML} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                This project is the collection of all assignments that were done
                during a University course (Machine Learning).
              </p>
              <p>
                "Assignments cover a series of topics in Machine Learning such
                as a deep learning framework implementation, a hidden markov
                model dynamic programming implementation and some
                implementations of evolutionary strategies (CEM, CMA-ES, NES).
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a href={"https://github.com/BrianPulfer/Machine-Learning"}>
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"AI7"} className={"project"}>
            <div>
              <p className="project-title">SmartBin (USI Hackathon 2019)</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"mx-auto prjimg"} src={SmartBin} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                This toy project took place during the USI Hackathon 2019 (48
                hours coding hackathon) event held by the University of Southern
                Switzerland. The goal of the hackathon was to use data provided
                by the city of Lugano and others (Swisscom, A.I.L, TPL, ...) to
                develop an application that would be beneficial for the city.
              </p>
              <p>
                The project that my team and I have come up with is a mock-up of
                how a smart trash bin would work and how easy would it be for it
                to classify correctly different trash types (Paper, ALU,
                Batteries, Plastic, Others), which would make life of humans
                easier and help climate.
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a href={"https://github.com/BrianPulfer/USI-Hackathon-2019"}>
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>

          <div id={"O1"} className={"project"}>
            <div>
              <p className="project-title">NannySearch</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={NannySearch} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                Tiny implementation of an information retrieval system (search
                engine).
              </p>
              <p>
                Apache Nutch and Solr were used to crawl and index a collection
                of around 1'000 web pages of british nannies. Spring boot was
                then used to create a tiny web application that would serve as
                interface for the user to the collection.
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  You can visit the project's repository at the following{" "}
                  <a href={"https://github.com/BrianPulfer/NannySearch"}>
                    link
                  </a>
                  .<br />A tiny documentation of the project can be found at the
                  following{" "}
                  <a
                    href={
                      process.env.PUBLIC_URL +
                      "/resources/docs/Brian Pulfer - NannySearch Report.pdf"
                    }
                  >
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
          <div id={"O2"} className={"project"}>
            <div>
              <p className="project-title">Tiforma Blockchain</p>
            </div>
            <Row className={"project-image"}>
              <Col className={"text-center"}>
                <Image className={"prjimg"} src={Tiforma} fluid />
              </Col>
            </Row>
            <div className={"project-description"}>
              <p>
                Semester project with the goal of implementing a web application
                based on blockchain for the management of students, courses,
                modules and more of the University (SUPSI).
              </p>
              <p>
                The goal of the project was to create a blockchain based
                back-end network through the hyperledger composer framework and
                provide users a front-end Angular7 interface to make transitions
                on the network. In the end, users were able to act the main CRUD
                operations on every entity of the network (students, courses,
                modules, classes, ...) as well as printing and browsing through
                the database.
              </p>
            </div>
            <div className={"project-links"}>
              <div className={"project-link"}>
                <p className={"text-center bold"}>
                  Source code is not available, but documentation of the project
                  can be found at the following{" "}
                  <a href={"https://github.com/gionasdev/tiforma-blockchain"}>
                    link
                  </a>
                  .
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Projects;
