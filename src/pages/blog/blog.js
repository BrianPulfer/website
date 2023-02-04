import React from "react";
import { Row, Col } from "react-bootstrap";
import { Image } from "react-bootstrap";

import "./blog.css";

import ppo from "./imgs/ppo.gif";
import vit from "./imgs/vit.png";
import ddpm from "./imgs/ddpm.gif";
import trackPage from "../../utilities/ga/ga";

class Blog extends React.Component {
  render() {
    trackPage();
    return (
      <div className={"blog-content"}>
        <div className="post">
          <p className="post-title">
            <a
              href={
                "https://medium.com/@brianpulfer/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b"
              }
            >
              PPO — Intuitive guide to state-of-the-art Reinforcement Learning
            </a>
          </p>
          <Row>
            <Col className={"text-center"}>
              <a
                href={
                  "https://medium.com/@brianpulfer/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b"
                }
              >
                <Image className={"prjimg"} src={ppo} fluid />
              </a>
            </Col>
          </Row>
          <p>
            I implemented the original 'Proximal Policy Optimization Algorithms'
            paper (Schulman et. al., 2017). I train a PPO model from scratch on
            the Cart-pole gym environment. Here you can access the{" "}
            <a href="https://medium.com/@brianpulfer/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b">
              medium story
            </a>
            . The code is publicly available.
          </p>
        </div>

        <div className="post">
          <p className="post-title">
            <a
              href={
                "https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1"
              }
            >
              Generating images with DDPMs: A PyTorch Implementation
            </a>
          </p>
          <Row>
            <Col className={"text-center"}>
              <a
                href={
                  "https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1"
                }
              >
                <Image className={"prjimg"} src={ddpm} fluid />
              </a>
            </Col>
          </Row>
          <p>
            I implemented the original{" "}
            <a href="https://arxiv.org/abs/2006.11239">
              <i>Denoising Diffusion Probabilistic Models</i> paper by Ho et.
              al.
            </a>{" "}
            I train a DDPM model from scratch on the MNIST and Fashion-MNIST
            datasets. The implementation comes with a{" "}
            <a href="https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1">
              medium story
            </a>
            . Code is publicly available.
          </p>
        </div>

        <div className="post">
          <p className="post-title">
            <a
              href={
                "https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c"
              }
            >
              Vision Transformers from Scratch
            </a>
          </p>
          <Row>
            <Col className={"text-center"}>
              <a
                href={
                  "https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c"
                }
              >
                <Image className={"prjimg"} src={vit} fluid />
              </a>
            </Col>
          </Row>
          <p>
            I implemented a simple version of Vision Transformers (ViT) from
            scratch in pytorch. Here's the link to read the full{" "}
            <a href="https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c">
              medium post
            </a>
            . Code is publicly available.
          </p>
        </div>
      </div>
    );
  }
}

export default Blog;
