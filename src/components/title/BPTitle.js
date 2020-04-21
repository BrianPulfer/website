import React from "react";

class BPTitle extends React.Component{
    constructor(props) {
        super(props);

        this.state = ({
            classes: props.classes,
            size: props.size,
            underline: props.underline,
            color: props.color,
            text: props.text,
            side: props.side
        });
    }

    render() {
        let classes = 'paragraph-center text-center display-4 bp-title';

        if(this.state.side){
            classes = 'paragraph-center display-4 bp-title';
        }

        const style = {
            "fontSize": this.state.size,
            "color" : this.state.color,
        };

        if(this.state.underline){
            style['borderBottom'] = ' solid black';
        }

        return (
            <div className={this.state.classes+' '+classes} style={style}>
                {this.state.text}
            </div>
        );
    }
}

export default BPTitle;