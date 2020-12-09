import React from "react";

import './BPParagraph.css'

class BPParagraph extends React.Component{
    constructor(props) {
        super(props);

        this.state = {
            classes: props.classes,
            text: props.text,
            size : props.size,
        };
    }

    render(){
        const style = {
            "fontSize" : this.state.size,
        };


        return (
            <p className={this.state.classes + " p bp-paragraph text-center"} style={style}>
                {this.state.text}
            </p>
        );
    }
}

export default BPParagraph