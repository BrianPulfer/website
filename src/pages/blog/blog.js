import React from 'react';

import './blog.css';
import Post from "../../components/post/post";

import vit from './imgs/vit.png';

class Blog extends React.Component {
    render() {
        return (
            <div className={"blog-content"}>
                <Post
                    title={'Vision Transformers from Scratch'}
                    img={vit}
                    description={"I implemented a simple version of Vision Transformers (ViT) from scratch in pytorch. Click on the above image to read the medium story."}
                    link={"https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c"}
                />
            </div>
        );
    }
}

export default Blog;