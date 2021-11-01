import React from 'react';

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import "./Publications.css";

class Publications extends React.Component {
    render() {
        const paragraphSize = "180%";

        return (
            <React.Fragment>
                <div className={"publication"}>
                    <BPTitle text={"No publication yet"} />
                </div>
            </React.Fragment>
        );
    }
}

export default Publications;