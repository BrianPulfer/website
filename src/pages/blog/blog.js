import React from 'react';

import './blog.css';
import Post from "../../components/post/post";

import vit from './imgs/vit.png';
import ddpm from './imgs/ddpm.gif';

class Blog extends React.Component {
    render() {
        return (
            <div className={"blog-content"}>
                <Post
                    title={'Generating images with DDPMs: A PyTorch Implementation'}
                    img={ddpm}
                    description={"I implemented the original 'Denoising Diffusion Probabilistic Models' paper by Ho et. al. I train a DDPM model from scratch on the MNIST and Fashion-MNIST datasets. Click on the above image to read the medium story. Code is publicly available."}
                    link={"https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1"}
                />

                <Post
                    title={'Vision Transformers from Scratch'}
                    img={vit}
                    description={"I implemented a simple version of Vision Transformers (ViT) from scratch in pytorch. Click on the above image to read the medium story. Code is publicly available."}
                    link={"https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c"}
                />
            </div>
        );
    }
}

export default Blog;