import React from 'react';

import {Col, Row} from "react-bootstrap";
import BPPublication from "../../components/publication/BPPublication";

import "./Publications.css";

class Publications extends React.Component {
    render() {
        return (
            <React.Fragment>
                <Row>
                    <Col className={'col-10 offset-1'}>
                        <div className={"publication"}>
                            <BPPublication
                                link={"https://arxiv.org/abs/2206.11793"}
                                date={"Jun 22"}
                                title={"Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach"}
                                abstract={"Copy detection patterns (CDP) are an attractive technology that allows manufacturers to defend their products against counterfeiting. The main assumption behind the protection mechanism of CDP is that these codes printed with the smallest symbol size (1x1) on an industrial printer cannot be copied or cloned with sufficient accuracy due to data processing inequality. However, previous works have shown that Machine Learning (ML) based attacks can produce high-quality fakes, resulting in decreased accuracy of authentication based on traditional feature-based authentication systems. While Deep Learning (DL) can be used as a part of the authentication system, to the best of our knowledge, none of the previous works has studied the performance of a DL-based authentication system against ML-based attacks on CDP with 1x1 symbol size. In this work, we study such a performance assuming a supervised learning (SL) setting."}
                                publishedon={"IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022."}
                                citation={"B. Pulfer, R. Chaban, Y. Belousov, J. Tutt, O. Taran, T. Holotyak, and S. Voloshynovskiy, “Authentication of copy detection patterns under machine learning attacks: A supervised approach,” in IEEE International Conference on Image Processing (ICIP), Bordeaux, France, October 2022."}
                                />
                        </div>
                    </Col>
                </Row>

                <Row>
                    <Col className={'col-10 offset-1'}>
                        <div className={"publication"}>
                            <BPPublication
                                link={"https://ieeexplore.ieee.org/document/9869302"}
                                date={"Dec 21"}
                                title={"Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems"}
                                abstract={"Safe deployment of self-driving cars (SDC) necessitates thorough simulated and in-field testing. Most testing techniques consider virtualized SDCs within a simulation environment, whereas less effort has been directed towards assessing whether such techniques transfer to and are effective with a physical real-world vehicle. In this paper, we leverage the Donkey Car open-source framework to empirically compare testing of SDCs when deployed on a physical small-scale vehicle vs its virtual simulated counterpart. In our empirical study, we investigate transferability of behavior and failure exposure between virtual and real-world environments on a vast set of corrupted and adversarial settings. While a large number of testing results do transfer between virtual and physical environments, we also identified critical shortcomings that contribute to the reality gap between the virtual and physical world, threatening the potential of existing testing solutions when applied to physical SDCs"}
                                publishedon={"IEEE Transactions on Software Engineering"}
                                citation={"A. Stocco, B. Pulfer and P. Tonella, \"Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems,\" in IEEE Transactions on Software Engineering, 2022, doi: 10.1109/TSE.2022.3202311."}/>
                        </div>
                    </Col>
                </Row>
            </React.Fragment>
        );
    }
}

export default Publications;