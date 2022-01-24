import React from "react";


import "./BPPublication.css";

class BPPublication extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            link: props.link,
            date: props.date,
            title: props.title,
            abstract: props.abstract,
            citation: props.citation
        }
    }

    render() {
        return (
            <div className={"publication"}>
                <h4><a href={this.state.link}>{this.state.title}</a> ({this.state.date})</h4>
                <b>Abstract:</b>
                <p>{this.state.abstract}</p>
                <i>{this.state.citation}</i>
            </div>
        );
    }
}

export default BPPublication;