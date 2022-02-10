import React from "react";

import {Row, Col} from "react-bootstrap";

import BPTitle from "../title/BPTitle";

import {
    TwitterTimelineEmbed,
    TwitterShareButton,
    TwitterFollowButton,
    TwitterHashtagButton,
    TwitterMentionButton,
    TwitterTweetEmbed,
    TwitterMomentShare,
    TwitterDMButton,
    TwitterVideoEmbed,
    TwitterOnAirButton
} from 'react-twitter-embed';

class Tweet extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            id: props.id,
            title: props.title,
            options: props.options
        };
    }

    render() {
        return (
            <div className={"tweet"}>
                <BPTitle text={this.state.title}/>
                <Row>
                    <Col/>
                    <Col>
                        <TwitterTweetEmbed tweetId={this.state.id} options={this.state.options}/>
                    </Col>
                    <Col/>
                </Row>
            </div>
        )
    }
}

export default Tweet;