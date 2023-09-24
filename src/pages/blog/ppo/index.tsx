import AppLayout from '@/components/Layout/AppLayout'
import CodeBlock from '@/components/Blog/CodeBlock'

import Head from 'next/head'
import BlogLayout from '../layout'
import { Center, Code, Image, Link, Text } from '@chakra-ui/react'

export default function PPO () {
  return (
        <>
            <Head><title>Blog - PPO</title></Head>
            <Text fontSize={'5xl'} textAlign={'center'}>PPO — Intuitive guide to state-of-the-art Reinforcement Learning</Text>
            <Center>
                <Link target="_blank" href="https://colab.research.google.com/drive/1u7YTohPaQFJPud8289pV6H65f9ZqSKWp?usp=sharing">
                    <Image src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
                </Link>
            </Center>

            <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Introduction</Text>
            <Text mb={5}>
                Proximal Policy Optimization (PPO) has been a state-of-the-art Reinforcement Learning (RL) algorithm since its proposal in the paper <Link href="https://arxiv.org/abs/1707.06347" textColor={'blue.500'}>Proximal Policy Optimization Algorithms (Schulman et. al., 2017)</Link>. This elegant algorithm can be and has been used for various tasks. Recently, it has also been used in the training of ChatGPT, the hottest machine-learning model at the moment.
            </Text>
            <Text mb={5}>
                PPO is not just widely used within the RL community, but it is also an excellent introduction to tackling RL through Deep Learning (DL) models.
            </Text>
            <Text mb={5}>
                In this article, I give a quick overview of the field of Reinforcement Learning, the taxonomy of algorithms to solve RL problems, and a review of the PPO algorithm proposed in the <Link href="https://arxiv.org/abs/1707.06347" textColor={'blue.500'}>paper</Link>. Finally, I share <Link href="https://colab.research.google.com/drive/1u7YTohPaQFJPud8289pV6H65f9ZqSKWp?usp=sharing" textColor={'blue.500'}>my own implementation</Link> of the PPO algorithm in PyTorch, comment on the obtained results and finish with a conclusion.
            </Text>

            <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Reinforcement Learning</Text>
            <Text mb={5}>
                The classical picture that is first shown to people approaching RL is the following:
            </Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/ppo/chatgpt.png" alt="ChatGPT explains RL"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    ChatGPT’s answer to the prompt: “Give an overview on the field of Reinforcement Learning”. While I asked help to ChatGPT for the introduction to the field of RL which was used to train ChatGPT itself (quite meta), I promise that everything in this article apart from this picture is written by me.
                </Text>
            </Center>

            <Text mb={5}>
                The classical picture that is first shown to people approaching RL is the following:
            </Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/ppo/rl.png" alt="Reinforcement learning setting"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Reinforcement Learning framework. Image from <Link href="https://neptune.ai/blog/reinforcement-learning-agents-training-debug">neptune.ai</Link>
                </Text>
            </Center>

            <Text mb={5}>
                At each timestamp, the environment provides the agent with a reward and an observation of the current state. Given this information, the agent takes an action in the environment which responds with a new reward and state and so on. This very general framework can be applied in a variety of domains.
            </Text>
            <Text mb={5}>
                Our goal is to create an agent that can maximize the obtained rewards. In particular, we are typically interested in maximizing the sum of discounted rewards
            </Text>
            <Center mb={5}>
                <Image src="/imgs/blog/ppo/reward.png" alt="Cumulative reward function"/>
            </Center>
            <Text mb={5}>
                Where γ is a discount factor typically in the range [0.95, 0.99], and r_t is the reward for timestamp t.
            </Text>

            <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Algorithms</Text>
            <Text mb={5}>
                So how do we solve an RL problem? There are multiple algorithms, but they can be divided (for Markov Decision Processes or MDPs) into two categories: <b>model-based</b> (create a model of the environment) and <b>model-free</b> (just learn what to do given a state).
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/taxonomy.png" alt="Taxonomy of RL methods"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                Taxonomy of Reinforcement Learning algorithms (from <Link href="https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html">OpenAI spinning up</Link>)
            </Text>
            </Center>
            <Text mb={5}>
                <b>Model-based</b> algorithms use a model of the environment and use this model to predict future states and rewards. The model is either given (e.g. a chessboard) or learned.
            </Text>
            <Text mb={5}>
                <b>Model-free</b> algorithms instead, directly learn how to act for the states encountered during training (Policy Optimization or PO), which state-action pairs yield good rewards (Q-Learning), or both at the same time.
            </Text>
            <Text mb={5}>
                <b>PPO</b> falls in the PO family of algorithms. We do not thus need a model of the environment to learn with the PPO algorithm. The main difference between PO and Q-Learning algorithms is that PO algorithms can be used in environments with continuous action space (i.e. where our actions have real values) and can find the optimal policy even if that policy is a stochastic one (i.e. acts probabilistically), whereas the Q-Learning algorithms cannot do either of those things. That’s one more reason to prefer PO algorithms. On the other hand, Q-Learning algorithms tend to be simpler, more intuitive, and nicer to train.
            </Text>
            <Text mb={5} fontSize={'xl'} fontWeight={'bold'}>
                Policy Optimization (Gradient-Based)
            </Text>
            <Text mb={5}>
                PO algorithms try to learn a policy directly. To do so, they either use gradient-free (e.g. genetic algorithms) or, perhaps more commonly, gradient-based algorithms.
            </Text>
            <Text mb={5}>
                By gradient-based methods, we refer to all methods that try to estimate the gradient of the learned policy with respect to the cumulative rewards. If we know this gradient (or an approximation of it), we can simply move the parameters of the policy toward the direction of the gradient to maximize rewards.
            </Text>

            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/grads.png" alt="Gradient-based RL methods"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Objective to be maximized with PO algorithms. Image from <Link href="https://lilianweng.github.io/posts/2018-04-08-policy-gradient/">Lil’Log’s blog</Link>.
                </Text>
            </Center>
            <Text mb={5}>
                Notice that there are multiple ways to estimate the gradient. Here we find listed 6 different values that we could pick as our maximization objective: the total reward, the reward following one action, the reward minus a baseline version, the state-action value function, the advantage function (used in the original PPO paper) and the temporal difference (TD) residual. In principle, they all provide an estimate of the real gradient we are interested in.
            </Text>

            <Text mb={5} fontSize={'xl'} fontWeight={'bold'}>
                PPO
            </Text>
            <Text mb={5}>
                <b>PPO</b> is a (model-free) Policy Optimization Gradient-based algorithm. The algorithm aims to learn a policy that maximizes the obtained cumulative rewards given the experience during training.
            </Text>
            <Text mb={5}>
                It is composed of an <b>actor πθ(. | st)</b> which outputs a probability distribution for the next action given the state at timestamp t, and by a <b>critic V(st)</b> which estimates the expected cumulative reward from that state (a scalar). Since both actor and critic take the state as an input, a backbone architecture can be shared between the two networks which extract high-level features.
            </Text>
            <Text mb={5}>
                PPO aims at making the policy more likely to select actions that have a high “advantage”, that is, that have a much higher measured cumulative reward than what the critic could predict. At the same time, we do not wish to update the policy too much in a single step, as it will probably incur in optimization problems. Finally, we would like to provide a bonus for the policy if it has high entropy, as to motivate exploration over exploitation.
            </Text>
            <Text mb={5}>
                The total loss function (to be maximized) is composed of three terms: a CLIP term, a Value Function (VF) term, and an entropy bonus.
            </Text>
            <Text mb={5}>
                The final objective is the following:
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/loss1.png" alt="PPO loss function"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    The loss function of PPO to be maximized.
                </Text>
            </Center>
            <Text mb={5}>
                Where c1 and c2 are hyper-parameters that weigh the importance of the accuracy of the critic and exploration capabilities of the policy respectively.
            </Text>
            <Text mb={5} fontSize={'l'} fontWeight={'bold'}>
                CLIP Term
            </Text>
            <Text mb={5}>
                The loss function motivates, as we said, the maximization of the probability of actions that resulted in an advantage (or minimization of the probability if the actions resulted in a negative advantage):
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/loss2.png" alt="PPO loss: first term"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    First loss term. We maximize the expected advantage while not moving the policy too much.
                </Text>
            </Center>
            <Text mb={5}>Where:</Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/ratio.png" alt="PPO loss ratio term"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Coefficient rt(θ). This is the term that gradients are going to go through.
                </Text>
            </Center>
            <Text mb={5}>
                Is a ratio that measures how likely we are to do that previous action now (with an updated policy) with respect to before. In principle, we do not wish this coefficient to be too high, as it means that the policy changed abruptly. That’s why we take the minimum of it and the clipped version between [1-ϵ, 1+ϵ], where ϵ is a hyper-parameter.
            </Text>
            <Text mb={5}>
            The advantage is computed as:
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/advantage.png" alt="Advantage function"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Advantage estimate. We simply take a difference between what we estimated the cumulative reward would have been given the initial state and the real cumulative reward observed up to a step t plus the estimate from that state onward. We apply a stop-gradient operator to this term in the CLIP loss.
                </Text>
            </Center>
            <Text mb={5}>
            We see that it simply measures how wrong the critic was for the given state st. If we obtained a higher cumulative reward, the advantage estimate will be positive and we will make the action we took in this state more likely. Vice-versa, if we expected a higher reward and we got a smaller one, the advantage estimate will be negative and we will make the action taken in this step less likely.
            </Text>
            <Text mb={5}>
            Notice that if we go all the way down to a state sT that was terminal, we do not need to rely on the critic itself and we can simply compare the critic with the actual cumulative reward. In that case, the estimate of the advantage is the true advantage. This is what we are going to do in our implementation of the cart-pole problem.
            </Text>
            <Text mb={5} fontSize={'l'} fontWeight={'bold'}>
                Value Function term
            </Text>
            <Text mb={5}>
            To have a good estimate of the advantage, however, we need a critic that can predict the value of a given state. This model is learned in a supervised fashion with a simple MSE loss:
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/loss3.png" alt="Loss term of the critic"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                The loss function for our critic is simply the Mean-Squared-Error between its predicted expected reward and the observed cumulative reward. We apply a stop-gradient operator only to the observed reward in this case and optimize the critic.
                </Text>
            </Center>
            <Text mb={5}>
                At each iteration, we update the critic too such that it will give us more and more accurate values for states as training progresses.
            </Text>
            <Text mb={5} fontSize={'l'} fontWeight={'bold'}>
                Entropy term
            </Text>
            <Text mb={5}>
                Finally, we encourage exploration with a small bonus on the entropy of the output distribution of the policy. We consider the standard entropy:
            </Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/entropy.png" alt="Entropy term in the loss of PPO"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Entropy formula for the output distribution given by the policy model.
                </Text>
            </Center>

            <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>
                Implementation
            </Text>
            <Text mb={5}>
                Don’t worry if the theory still seems a bit shady. The implementation will hopefully make everything clear.
            </Text>
            <Text mb={5} fontSize={'xl'} fontWeight={'bold'}>
                PPO-independent Code
            </Text>
            <Text mb={5}>
                Let’s start with the imports:
            </Text>
            <CodeBlock language="python">
                {
`from argparse import ArgumentParser

import gym
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl`}
            </CodeBlock>
            <Text mb={5}>
                The important hyper-parameters of PPO are the number of actors, horizon, epsilon, the number of epochs for each optimization phase, the learning rate, the discount factor gamma, and the constants that weigh the different loss terms c1 and c2. We collect these through the program arguments.
            </Text>
            <CodeBlock language="python">{
`def parse_args():
"""Pareser program arguments"""
# Parser
parser = ArgumentParser()

# Program arguments (default for Atari games)
parser.add_argument("--max_iterations", type=int, help="Number of iterations of training", default=100)
parser.add_argument("--n_actors", type=int, help="Number of actors for each update", default=8)
parser.add_argument("--horizon", type=int, help="Number of timestamps for each actor", default=128)
parser.add_argument("--epsilon", type=float, help="Epsilon parameter", default=0.1)
parser.add_argument("--n_epochs", type=int, help="Number of training epochs per iteration", default=3)
parser.add_argument("--batch_size", type=int, help="Batch size", default=32 * 8)
parser.add_argument("--lr", type=float, help="Learning rate", default=2.5 * 1e-4)
parser.add_argument("--gamma", type=float, help="Discount factor gamma", default=0.99)
parser.add_argument("--c1", type=float, help="Weight for the value function in the loss function", default=1)
parser.add_argument("--c2", type=float, help="Weight for the entropy bonus in the loss function", default=0.01)
parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=5)
parser.add_argument("--seed", type=int, help="Randomizing seed for the experiment", default=0)

# Dictionary with program arguments
return vars(parser.parse_args())
`}
            </CodeBlock>
            <Text mb={5}>
                Notice that, by default, the parameters are set as described in the paper. Ideally, our code should run on GPU if possible, so we create a simple utility function.
            </Text>
            <CodeBlock language="python">{
`def get_device():
    """Gets the device (GPU if any) and logs the type"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found GPU device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU found: Running on CPU")
    return device
`}
            </CodeBlock>
            <Text mb={5}>
                When we apply RL, we typically have a buffer that stores states, actions, and rewards encountered by the current model. These are used to update our models. We create a utility function <Code>run_timestamps</Code> that will run a given model on a given environment for a fixed number of timestamps (re-setting the environment if the episode finishes). We also use an option <Code>render=False</Code> in case we just want to see how the trained model does.
            </Text>
            <CodeBlock language="python">{
`@torch.no_grad()
def run_timestamps(env, model, timestamps=128, render=False, device="cpu"):
    """Runs the given policy on the given environment for the given amount of timestamps.
     Returns a buffer with state action transitions and rewards."""
    buffer = []
    state = env.reset()[0]

    # Running timestamps and collecting state, actions, rewards and terminations
    for ts in range(timestamps):
        # Taking a step into the environment
        model_input = torch.from_numpy(state).unsqueeze(0).to(device).float()
        action, action_logits, value = model(model_input)
        new_state, reward, terminated, truncated, info = env.step(action.item())

        # Rendering / storing (s, a, r, t) in the buffer
        if render:
            env.render()
        else:
            buffer.append([model_input, action, action_logits, value, reward, terminated or truncated])

        # Updating current state
        state = new_state

        # Resetting environment if episode terminated or truncated
        if terminated or truncated:
            state = env.reset()[0]

    return buffer
`}
            </CodeBlock>
            <Text mb={5}>
                The output of the function (when not rendering) is a buffer containing states, taken actions, action probabilities (logits), estimated critic’s values, rewards, and the termination state for the provided policy for each timestamp. Notice that the function uses the decorator <b>@torch.no_grad()</b>, so we will not need to store gradients for the actions taken during the interactions with the environment.
            </Text>

            <Text mb={5} fontSize={'xl'} fontWeight={'bold'}>
                Code for PPO
            </Text>
            <Text mb={5}>
                Now that we got the trivial stuff out of the way, is time to implement the core algorithm.
            </Text>
            <Text mb={5}>
                Ideally, we would like our <b>main</b> function to look something like this:
            </Text>
            <CodeBlock language="python">{
`def main():
    # Parsing program arguments
    args = parse_args()
    print(args)

    # Setting seed
    pl.seed_everything(args["seed"])

    # Getting device
    device = get_device()

    # Creating environment (discrete action space)
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # Creating the model, training it and rendering the result
    # (We are missing this part 😅)
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)
    training_loop(env, model, args)
    model = load_best_model()
    testing_loop(env, model)
`}
            </CodeBlock>
            <Text mb={5}>
                We already got most of it. We just need to define the PPO model, the training, and the test functions.
            </Text>
            <Text mb={5}>
                The architecture of the PPO model is not the interesting part here. We just need two models (actor and critic) that will act in the environment. Of course, the model architecture plays a crucial role in harder tasks, but with the cart pole, we can be confident that some MLP will do the job.
            </Text>
            <Text mb={5}>
                Thus, we can create a MyPPO class that contains actor and critic models. Optionally, we may decide that part of the architecture between the two is shared. When running the forward method for some states, we return the sampled actions by the actor, the relative probabilities for each possible action (logits), and the critic’s estimated values for each state.
            </Text>
            <CodeBlock language="python">{
`class MyPPO(nn.Module):
"""Implementation of a PPO model. The same backbone is used to get actor and critic values."""

    def __init__(self, in_shape, n_actions, hidden_d=100, share_backbone=False):
        # Super constructor
        super(MyPPO, self).__init__()

        # Attributes
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

        # Shared backbone for policy and value functions
        in_dim = np.prod(in_shape)

        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )

        self.backbone = to_features() if self.share_backbone else nn.Identity()

        # State action function
        self.actor = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value function
        self.critic = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        action = self.actor(features)
        value = self.critic(features)
        return Categorical(action).sample(), action, value
`}
            </CodeBlock>
            <Text mb={5}>
                Notice that <Code>Categorical(action).sample()</Code> creates a categorical distribution with the action logits and samples from it one action (for each state).
            </Text>

            <Text mb={5}>Finally, we can take care of the actual algorithm in the training_loop function. As we know from the paper, the actual signature of the function should look something like this:</Text>
            <CodeBlock language="python">{
`def training_loop(env, model, max_iterations, n_actors, horizon, gamma, 
    epsilon, n_epochs, batch_size, lr, c1, c2, device, env_name=""):
    # TODO...
`}
            </CodeBlock>
            <Text mb={5}>{'Here’s'} the pseudo-code provided in the paper for the PPO training procedure:</Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/pseudocode.png" alt="Pseudocode for PPO"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Pseudo code for PPO training provided in the <Link href="https://arxiv.org/abs/1707.06347">original paper</Link>.
                </Text>
            </Center>
            <Text mb={5}>The pseudo-code for PPO is relatively simple: we simply collect interactions with the environment by multiple copies of our policy model (called actors) and use the objective previously defined to optimize both actor and critic networks.</Text>
            <Text mb={5}>Since we need to measure the cumulative rewards that we really obtained, we create a function that, given a buffer, replaces rewards at each timestamp with the cumulative rewards:</Text>
            <CodeBlock language="python">{
`def compute_cumulative_rewards(buffer, gamma):
    """Given a buffer with states, policy action logits, rewards and terminations,
    computes the cumulative rewards for each timestamp and substitutes them into the buffer."""
    curr_rew = 0.

    # Traversing the buffer on the reverse direction
    for i in range(len(buffer) - 1, -1, -1):
        r, t = buffer[i][-2], buffer[i][-1]

        if t:
            curr_rew = 0
        else:
            curr_rew = r + gamma * curr_rew

        buffer[i][-2] = curr_rew

    # Getting the average reward before normalizing (for logging and checkpointing)
    avg_rew = np.mean([buffer[i][-2] for i in range(len(buffer))])

    # Normalizing cumulative rewards
    mean = np.mean([buffer[i][-2] for i in range(len(buffer))])
    std = np.std([buffer[i][-2] for i in range(len(buffer))]) + 1e-6
    for i in range(len(buffer)):
        buffer[i][-2] = (buffer[i][-2] - mean) / std

    return avg_rew
`}
            </CodeBlock>
            <Text mb={5}>Notice that, in the end, we normalize the cumulative rewards. This is a standard trick to make the optimization problem easier and the training smoother.</Text>
            <Text mb={5}>Now that we can obtain a buffer with states, actions taken, actions probabilities, and cumulative rewards, we can write a function that, given a buffer, computes the three loss terms for our final objective:</Text>
            <CodeBlock language="python">{
`def get_losses(model, batch, epsilon, annealing, device="cpu"):
    """Returns the three loss terms for a given model and a given batch and additional parameters"""
    # Getting old data
    n = len(batch)
    states = torch.cat([batch[i][0] for i in range(n)])
    actions = torch.cat([batch[i][1] for i in range(n)]).view(n, 1)
    logits = torch.cat([batch[i][2] for i in range(n)])
    values = torch.cat([batch[i][3] for i in range(n)])
    cumulative_rewards = torch.tensor([batch[i][-2] for i in range(n)]).view(-1, 1).float().to(device)

    # Computing predictions with the new model
    _, new_logits, new_values = model(states)

    # Loss on the state-action-function / actor (L_CLIP)
    advantages = cumulative_rewards - values
    margin = epsilon * annealing
    ratios = new_logits.gather(1, actions) / logits.gather(1, actions)

    l_clip = torch.mean(
        torch.min(
            torch.cat(
                (ratios * advantages,
                torch.clip(ratios, 1 - margin, 1 + margin) * advantages),
                dim=1),
            dim=1
        ).values
    )

    # Loss on the value-function / critic (L_VF)
    l_vf = torch.mean((cumulative_rewards - new_values) ** 2)

    # Bonus for entropy of the actor
    entropy_bonus = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-5)), dim=1))

    return l_clip, l_vf, entropy_bonus
`}
            </CodeBlock>
            <Text mb={5}>Notice that, in practice, we use an <Code>annealing</Code> parameter that is set to 1 and linearly decayed towards 0 throughout the training. The idea is that as training progresses, we want our policy to change less and less. Also notice that the <Code>advantages</Code> variable is a simple difference between tensors for which we are not tracking gradients, unlike <Code>new_logits</Code> and <Code>new_values</Code>.</Text>
            <Text mb={5}>Now that we have a way to interact with the environment and store buffers, compute the (true) cumulative rewards and obtain the loss terms, we can write the final training loop:</Text>
            <CodeBlock language="python">{
`def training_loop(env, model, max_iterations, n_actors, horizon, gamma, epsilon, n_epochs, batch_size, lr,
    c1, c2, device, env_name=""):
    """Train the model on the given environment using multiple actors acting up to n timestamps."""

    # Starting a new Weights & Biases run
    wandb.init(project="Papers Re-implementations",
    entity="peutlefaire",
    name=f"PPO - {env_name}",
    config={
        "env": str(env),
        "number of actors": n_actors,
        "horizon": horizon,
        "gamma": gamma,
        "epsilon": epsilon,
        "epochs": n_epochs,
        "batch size": batch_size,
        "learning rate": lr,
        "c1": c1,
        "c2": c2
    })

    # Training variables
    max_reward = float("-inf")
    optimizer = Adam(model.parameters(), lr=lr, maximize=True)
    scheduler = LinearLR(optimizer, 1, 0, max_iterations * n_epochs)
    anneals = np.linspace(1, 0, max_iterations)

    # Training loop
    for iteration in range(max_iterations):
    buffer = []
    annealing = anneals[iteration]

    # Collecting timestamps for all actors with the current policy
    for actor in range(1, n_actors + 1):
    buffer.extend(run_timestamps(env, model, horizon, False, device))

    # Computing cumulative rewards and shuffling the buffer
    avg_rew = compute_cumulative_rewards(buffer, gamma)
    np.random.shuffle(buffer)

    # Running optimization for a few epochs
    for epoch in range(n_epochs):
    for batch_idx in range(len(buffer) // batch_size):
    # Getting batch for this buffer
    start = batch_size * batch_idx
    end = start + batch_size if start + batch_size < len(buffer) else -1
    batch = buffer[start:end]

    # Zero-ing optimizers gradients
    optimizer.zero_grad()

    # Getting the losses
    l_clip, l_vf, entropy_bonus = get_losses(model, batch, epsilon, annealing, device)

    # Computing total loss and back-propagating it
    loss = l_clip - c1 * l_vf + c2 * entropy_bonus
    loss.backward()

    # Optimizing
    optimizer.step()
    scheduler.step()

    # Logging information to stdout
    curr_loss = loss.item()
    log = f"Iteration {iteration + 1} / {max_iterations}: " \
    f"Average Reward: {avg_rew:.2f}\t" \
    f"Loss: {curr_loss:.3f} " \
    f"(L_CLIP: {l_clip.item():.1f} | L_VF: {l_vf.item():.1f} | L_bonus: {entropy_bonus.item():.1f})"
    if avg_rew > max_reward:
    torch.save(model.state_dict(), MODEL_PATH)
    max_reward = avg_rew
    log += " --> Stored model with highest average reward"
    print(log)

    # Logging information to W&B
    wandb.log({
    "loss (total)": curr_loss,
    "loss (clip)": l_clip.item(),
    "loss (vf)": l_vf.item(),
    "loss (entropy bonus)": entropy_bonus.item(),
    "average reward": avg_rew
    })

    # Finishing W&B session
    wandb.finish()          
`}
            </CodeBlock>
            <Text mb={5}>Finally, to see how the final model does, we use the following testing_loop function:</Text>
            <CodeBlock language="python">{
`def testing_loop(env, model, n_episodes, device):
    """Runs the learned policy on the environment for n episodes"""
    for _ in range(n_episodes):
        run_timestamps(env, model, timestamps=128, render=True, device=device)
`}
            </CodeBlock>
            <Text mb={5}>And our main program is simply:</Text>
            <CodeBlock language="python">{
`def main():
    # Parsing program arguments
    args = parse_args()
    print(args)

    # Setting seed
    pl.seed_everything(args["seed"])

    # Getting device
    device = get_device()

    # Creating environment (discrete action space)
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # Creating the model (both actor and critic)
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)

    # Training
    training_loop(env, model, args["max_iterations"], args["n_actors"], args["horizon"], args["gamma"], args["epsilon"],
                args["n_epochs"], args["batch_size"], args["lr"], args["c1"], args["c2"], device, env_name)

    # Loading best model
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Testing
    env = gym.make(env_name, render_mode="human")
    testing_loop(env, model, args["n_test_episodes"], device)
    env.close()
`}
            </CodeBlock>
            <Text mb={5}>And that’s all for the implementation! If you made it this far, congratulations. You pro now know how to implement the PPO algorithm.</Text>

            <Text mb={5} fontSize={'3xl'} fontWeight={'bold'}>Results</Text>
            <Text mb={5}>The Weights & Biases logs allow us to visualize the logged metrics and losses. In particular, we have access to plots of the loss and its terms and the average reward per iteration.</Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/lossess.png" alt="Total loss"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Training losses through training iterations. The total loss (blue) is the sum of L_CLIP (orange) minus the L_VF (pink) plus a small constant times the entropy bonus (green)
                </Text>
            </Center>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/avgreward.png" alt="Average reward"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Average reward through iterations. PPO quickly learns to maximize the cumulative reward.
                </Text>
            </Center>
            <Text mb={5}>As the cart pole environment is not extremely challenging, our algorithm quickly finds a solution to the problem, maximizing the average reward after just ~20 steps. Also, since the environment only has 2 possible actions, the entropy term remains basically fixed.</Text>
            <Text mb={5}>Finally, here’s what we get if we render the final policy in action!</Text>
            <Center className="flex flex-col" mb={5}>
                <Image src="/imgs/blog/ppo/ppo.gif" alt="Balancing cart-pole with PPO"/>
                <Text textAlign={'center'} textColor={'gray.500'} fontSize={'sm'}>
                    Trained PPO model balancing the cart pole
                </Text>
            </Center>

            <Text mb={5} fontSize={'3xl'} fontWeight={'bold'}>Conclusions</Text>
            <Text mb={5}>PPO is a state-of-the-art RL policy optimization (thus model-free) algorithm and as such, it can be virtually used in any environment. Also, PPO has a relatively simple objective function and relatively few hyper-parameters to be tuned.</Text>
            <Text mb={5}>If you would like to play with the algorithm on the fly, here’s a link to the <Link textColor={'blue.500'} href="https://colab.research.google.com/drive/1u7YTohPaQFJPud8289pV6H65f9ZqSKWp?usp=sharing">Colab Notebook</Link>. You can find my personal up-to-date re-implementation of the PPO algorithm (as a .py file) under the <Link textColor={'blue.500'} href="https://github.com/BrianPulfer/PapersReimplementations">GitHub repository</Link>. Feel free to play around with it or adapt it to your own project!</Text>
            <Text mb={5}>If you enjoyed this story, let me know! Feel free to reach out for further discussions. Wish you happy hacking with PPO ✌️</Text>
        </>
  )
}

PPO.getLayout = function getLayout (page: React.ReactElement) {
  return (
        <AppLayout>
            <BlogLayout>
                {page}
            </BlogLayout>
        </AppLayout>
  )
}
