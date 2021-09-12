import React from 'react';

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import "./Publications.css";

class Publications extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        const paragraphSize = "180%";

        return (
            <React.Fragment>
                <div className={"publication"}>
                    <BPTitle text={"No publication yet 😅"} />
                    <BPParagraph
                        size={paragraphSize}
                        text={"As I am starting my Ph.D. on Nov 1st, I have still no official publication except for my BSc and MSc thesis, which you can find under the 'Projects' section."}
                    />
                </div>
            </React.Fragment>
        );
    }
}

export default Publications;