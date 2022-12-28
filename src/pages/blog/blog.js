import React from 'react';
import ReactGA from "react-ga";

import './blog.css';
import Post from "../../components/post/post";

import ppo from "./imgs/ppo.gif";
import vit from './imgs/vit.png';
import ddpm from './imgs/ddpm.gif';

class Blog extends React.Component {
    componentDidMount(){
        ReactGA.initialize('G-BH82F18037');
        ReactGA.pageview(window.location.pathname + window.location.hash);
    }

    render() {
        return (
            <div className={"blog-content"}>
                <Post
                    title={'PPO — Intuitive guide to state-of-the-art Reinforcement Learning'}
                    img={ppo}
                    description={"I implemented the original 'Proximal Policy Optimization Algorithms' paper (Schulman et. al., 2017). I train a PPO model from scratch on the Cart-pole gym environment. Click on the above image to read the medium story. Code is publicly available."}
                    link={"https://medium.com/@brianpulfer/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b"}
                />
                
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