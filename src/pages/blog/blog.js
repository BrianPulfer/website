import React from 'react';

import './blog.css';
import Tweet from "../../components/tweet/tweet";

class Blog extends React.Component {
    constructor(props) {
        super(props);
    }


    render() {
        return (
            <div className={"blog-content"}>
                <Tweet id={'1489272716918235145'} />
            </div>
        );
    }
}

export default Blog;