import React from "react";

import { Row, Col } from "react-bootstrap";

import BPTitle from "../title/BPTitle";
import { Image } from "react-bootstrap";
import "./post.css";
import BPParagraph from "../paragraph/BPParagraph";

class Post extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      title: props.title,
      img: props.img,
      description: props.description,
      link: props.link,
    };
  }

  render() {
    return (
      <div className={"BPost"}>
        <a href={this.state.link}>
          <BPTitle text={this.state.title} />
        </a>
        <Row>
          <Col className={"text-center"}>
            <a href={this.state.link}>
              <Image className={"prjimg"} src={this.state.img} fluid />
            </a>
          </Col>
        </Row>
        <BPParagraph text={this.state.description} size={24} />
      </div>
    );
  }
}

export default Post;
