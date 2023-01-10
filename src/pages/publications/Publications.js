import React from 'react';

import {Col, Row} from "react-bootstrap";
import BPPublication from "../../components/publication/BPPublication";

import "./Publications.css";
import trackPage from "../../utilities/ga/ga";

class Publications extends React.Component {
    render() {
        trackPage();
        return (
            <React.Fragment>
                <Row>
                    <Col className={'col-10 offset-1'}>
                        <BPPublication
                            link={"https://arxiv.org/abs/2212.02456"}
                            date={"Dec 22"}
                            title={"Solving the Weather4cast Challenge via Visual Transformers for 3D Images"}
                            abstract={"Accurately forecasting the weather is an important task, as many real-world processes and decisions depend on future meteorological conditions. The NeurIPS 2022 challenge entitled Weather4cast poses the problem of predicting rainfall events for the next eight hours given the preceding hour of satellite observations as a context. Motivated by the recent success of transformer-based architectures in computer vision, we implement and propose two methodologies based on this architecture to tackle this challenge. We find that ensembling different transformers with some baseline models achieves the best performance we could measure on the unseen test data. Our approach has been ranked 3rd in the competition."}
                            publishedon={"ArXiv (NeurIPS 2022 challenge), 2022"}
                            citation={"Belousov, Y., Polezhaev, S., Pulfer, B."}
                        />
                    </Col>
                </Row>

                <Row>
                    <Col className={'col-10 offset-1'}>
                        <BPPublication
                            link={"https://arxiv.org/abs/2209.15625"}
                            date={"Aug 22"}
                            title={"Anomaly localization for copy detection patterns through print estimation"}
                            abstract={"Copy detection patterns (CDP) are recent technologies for protecting products from counterfeiting. However, in contrast to traditional copy fakes, deep learning-based fakes have shown to be hardly distinguishable from originals by traditional authentication systems. Systems based on classical supervised learning and digital templates assume knowledge of fake CDP at training time and cannot generalize to unseen types of fakes. Authentication based on printed copies of originals is an alternative that yields better results even for unseen fakes and simple authentication metrics but comes at the impractical cost of acquisition and storage of printed copies. In this work, to overcome these shortcomings, we design a machine learning (ML) based authentication system that only requires digital templates and printed original CDP for training, whereas authentication is based solely on digital templates, which are used to estimate original printed codes. The obtained results show that the proposed system can efficiently authenticate original and detect fake CDP by accurately locating the anomalies in the fake CDP. The empirical evaluation of the authentication system under investigation is performed on the original and ML-based fakes CDP printed on two industrial printers."}
                            publishedon={"IEEE International Workshop on Information Forensics and Security (WIFS), 2022"}
                            citation={"Pulfer, B., Belousov, Y., Tutt, J., Chaban, R., Taran, O., Holotyak, T., & Voloshynovskiy, S.. (2022). \"Anomaly localization for copy detection patterns through print estimations,\" in IEEE International Workshop on Information Forensics and Security (WIFS), 2022"}
                        />
                    </Col>
                </Row>

                <Row>
                    <Col className={'col-10 offset-1'}>
                        <BPPublication
                            link={"https://arxiv.org/abs/2206.11793"}
                            date={"Jun 22"}
                            title={"Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach"}
                            abstract={"Copy detection patterns (CDP) are an attractive technology that allows manufacturers to defend their products against counterfeiting. The main assumption behind the protection mechanism of CDP is that these codes printed with the smallest symbol size (1x1) on an industrial printer cannot be copied or cloned with sufficient accuracy due to data processing inequality. However, previous works have shown that Machine Learning (ML) based attacks can produce high-quality fakes, resulting in decreased accuracy of authentication based on traditional feature-based authentication systems. While Deep Learning (DL) can be used as a part of the authentication system, to the best of our knowledge, none of the previous works has studied the performance of a DL-based authentication system against ML-based attacks on CDP with 1x1 symbol size. In this work, we study such a performance assuming a supervised learning (SL) setting."}
                            publishedon={"IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022."}
                            citation={"B. Pulfer, R. Chaban, Y. Belousov, J. Tutt, O. Taran, T. Holotyak, and S. Voloshynovskiy, “Authentication of copy detection patterns under machine learning attacks: A supervised approach,” in IEEE International Conference on Image Processing (ICIP), Bordeaux, France, October 2022."}
                        />
                    </Col>
                </Row>

                <Row>
                    <Col className={'col-10 offset-1'}>
                        <BPPublication
                            link={"https://ieeexplore.ieee.org/document/9869302"}
                            date={"Dec 21"}
                            title={"Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems"}
                            abstract={"Safe deployment of self-driving cars (SDC) necessitates thorough simulated and in-field testing. Most testing techniques consider virtualized SDCs within a simulation environment, whereas less effort has been directed towards assessing whether such techniques transfer to and are effective with a physical real-world vehicle. In this paper, we leverage the Donkey Car open-source framework to empirically compare testing of SDCs when deployed on a physical small-scale vehicle vs its virtual simulated counterpart. In our empirical study, we investigate transferability of behavior and failure exposure between virtual and real-world environments on a vast set of corrupted and adversarial settings. While a large number of testing results do transfer between virtual and physical environments, we also identified critical shortcomings that contribute to the reality gap between the virtual and physical world, threatening the potential of existing testing solutions when applied to physical SDCs"}
                            publishedon={"IEEE Transactions on Software Engineering"}
                            citation={"A. Stocco, B. Pulfer and P. Tonella, \"Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems,\" in IEEE Transactions on Software Engineering, 2022, doi: 10.1109/TSE.2022.3202311."}/>
                    </Col>
                </Row>
            </React.Fragment>
        );
    }
}

export default Publications;