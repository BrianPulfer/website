import AppLayout from '@/components/Layout/AppLayout'
import BlogLayout from '../layout'
import Head from 'next/head'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'
import CodeBlock from '@/components/Blog/CodeBlock'
import { Center, Code, Image, Link, Text } from '@chakra-ui/react'

export default function IGN (): JSX.Element {
  return (
    <>
      <Head><title>Blog - IGN</title></Head>
      <Text fontSize={'l'} textAlign={'right'}><b>Published:</b> 18.12.2023</Text>
      <Text fontSize={'5xl'} textAlign={'center'}>Idempotent Generative Networks</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Introduction</Text>
      <Text mb={5}>Recently, I {"don't"} really recall how / by which medium, I stepped upon <Link textColor={'blue.500'} href='https://arxiv.org/abs/2311.01462'>Idempotent Generative Network</Link> by <i>Shocher et. al.</i> The paper was very well written, presented a simple yet novel and elegant idea and kicked off with a funny quote:</Text>

      <Center>
        <Text fontSize={'xl'}>
            GEORGE: <i>{"You're gonna”overdry”it."}</i><br/>
            JERRY: <i>{"You, you can't ”overdry.”"}</i><br/>
            GEORGE: <i>{'"Why not?"'}</i><br/>
            JERRY: <i>{"Same as you can't ”overwet.” You see, once something is wet, it's wet. Same thing with dead: like once you die you're dead, right? Let's say you drop dead and I shoot you: you're not gonna die again, you're already dead. You can't ”overdie,” you can't ”overdry.”"}</i>
        </Text>
      </Center>
      <Text mb={5} fontSize={'xl'} align={'right'}>{'"Seinfield"'}, Season 1, Episode 1, NBC 1989</Text>
      <Text mb={5}>Underneath the irony of this quote, there is an interesting idea: Some concepts are binary by nature. Something is either wet or dry. Someone is either dead or alive (<i><Link textColor={'blue.500'} href='https://simple.wikipedia.org/wiki/Schr%C3%B6dinger%27s_cat'>{"Schrödinger's cat"}</Link> laughs in super-position</i>).
       A sample is either in-distribution or out of distribution. So functions that bring this binary state to one of the two ends, like drying, are idempotent. If you dry something, it gets dry. If you dry it again, it will stay dry just as it was.</Text>

       <Text mb={5}>Driven by this last observation, the question (in the context of generative models) is: <i>What if we learned a model that could {'"in-distribufy"'}?</i> In other words, what if we enforced the fact that a prior distribution (random noise) has to be mapped to our data distribution, but things in-distribution cannot be made yet more in-distribution?</Text>
       <Text mb={5}>Then, we could draw something like this: </Text>

      <Center mb={5} className="flex flex-col">
        <Image className="flex flex-row" src="/imgs/blog/ign/ign.png" alt="Idempotent Generative Network" />
        <Text textAlign={'center'} textColor={'gray.500'} fontStyle={'italic'} fontSize={'sm'}>
            Figure 1 from <Link textColor={'blue.500'} href='https://arxiv.org/abs/2311.01462'>Idempotent Generative Network</Link> by <i>Shocher et. al.</i>
        </Text>
      </Center>

      <Text mb={5}>where we have samples <InlineMath math='\mathbf{z}'/> from a source distribution that have to be put in-distribution with a function <InlineMath math='f_{\theta}'/>, but <InlineMath math='f_{\theta}'/> has to leave in-distribution samples <InlineMath math='\mathbf{x}'/> untouched.
      This gives rise to two objectives:</Text>

      <Center>
        <BlockMath math={String.raw`f_{\theta}(\mathbf{x}) = \mathbf{x}`}/>
        <Text ml={5}>Reconstruction term</Text>
      </Center>
      <Center>
        <BlockMath math={String.raw`f_{\theta}(f_{\theta}(\mathbf{z})) = f_{\theta}(\mathbf{z})`}/>
        <Text ml={5}>Idempotent term</Text>
      </Center>

      <Text mb={5}>where with the first, we encourage the function to give us back the same sample for in-disitribution samples, and with the second we encourage the function to be idempotent, that is, applying it twice should give the same effect as applying it just once (like when drying).</Text>
      <Text mb={5}>There is still something missing though: both objectives are perfectly satisfied if <InlineMath math='f_\theta'/> is the identity function. In fact, the identity function is the most basic idempotent function there is, and as of right now we are only optimizing for idempotence.</Text>
      <Text mb={5}>This is where the authors make the key observation that there are two pathways for the gradient: one, which is the desired one, encourages <InlineMath math='f_\theta(\mathbf{z})'/> to put things into target distribution. In fact, if <InlineMath math='f_\theta(\mathbf{z}) \approx \mathbf{x}'/>, since we already encourage <InlineMath math='f_\theta(\mathbf{x}) = \mathbf{x}'/> we automatically have that <InlineMath math='f_{\theta}(f_{\theta}(\mathbf{z})) = f_{\theta}(\mathbf{z})'/>.
      The other pathway instead, which we want to avoid, encourages <InlineMath math='f_\theta'/> to act as an identity regardless if the given input is in distribution or not. This is well represented with the second figure from the paper:</Text>

      <Center mb={5} className="flex flex-col">
        <Image className="flex flex-row" src="/imgs/blog/ign/grads.png" alt="Pathways of gradients" />
        <Text textAlign={'center'} textColor={'gray.500'} fontStyle={'italic'} fontSize={'sm'}>
            Figure 2 from <Link textColor={'blue.500'} href='https://arxiv.org/abs/2311.01462'>Idempotent Generative Network</Link> by <i>Shocher et. al.</i>
        </Text>
      </Center>

      <Text mb={5}>In the figure, we see a data manifold in blue and the prior distribution as the sphere / circle. The red pathway <InlineMath math='\color{red}{\Delta f}'/> pushes <InlineMath math='f_\theta(\mathbf{z})'/> to be mapped onto the data manifold, where <InlineMath math='f_\theta(f_\theta(\mathbf{z}))=f_\theta(\mathbf{z})'/> will also be enforced by <InlineMath math='f_\theta(\mathbf{x}) = \mathbf{x}'/>.
      The green pathway <InlineMath math='\color{green}{\Delta f}'/> is instead trying to map back to whatever we got in the first run, whether it was on the data manifold or not.</Text>

      <Text mb={5}>So the key idea now is to favour the good pathway, while discouraging the bad one. In this way, we get the desired behaviour out of the model by tightening the data manifold! This gives rise to a third term:</Text>

      <Center>
        <BlockMath math={String.raw`f_{\theta}(f_{\theta}(\mathbf{z})) \ne f_{\theta}(\mathbf{z})`}/>
        <Text ml={5}>Tightening term</Text>
      </Center>

      <Text mb={5}>Which is the exact opposite of the idempotent term ...Wait, what?</Text>
      <Text mb={5}>Yes, we are in fact optimizing two exact opposing things. We want to make the function idempotent, but also to make it the exact opposite of idempotent. However, {"here's"} the catch: We want the function to be idempotent only within the data manifold, and to be quite the exact opposite of idempotent outside!</Text>
      <Text mb={5}>To do so, we encourage the model to output something that will remain the same whether it is mapped through a copy of the model again or not. At the same time, we encourage the model to map something that already went through the model once to something as different as possible.</Text>

      <Text mb={5}>This is perhaps better explained with an example. {"Let's"} assume that <InlineMath math='f_\theta(\mathbf{z})'/> maps onto the data manifold. Then, despite the fact that we have one term trying to make <InlineMath math='f_\theta(f_\theta(\mathbf{z})) \ne f_\theta(\mathbf{z})'/>, we also have two terms opposing this effect (the term on the inner part and the reconstruction term). Our function will act idempotent inside the data manifold.
      {"Let's"} now assume that <InlineMath math='f_\theta(\mathbf{z})'/> maps out of distribution. We now have only one term fighting this effect, since the reconstruction term only works on the data manifold. Our function will thus try to map things closer to the data manifold.</Text>

      <Text mb={5}>Here is the total loss function:</Text>

      <BlockMath>{'\\mathcal{L}_{\\text{rec}}(\\theta) = \\mathbb{E}_\\mathbf{x} [ D(f_\\theta(\\mathbf{x}), \\mathbf{x})] '}</BlockMath>
      <BlockMath>{'\\mathcal{L}_{\\text{idem}}(\\mathbf{z}; \\theta, \\hat{\\theta}) = D(f_{\\hat{\\theta}}(f_\\theta(\\mathbf{z})), f_\\theta(\\mathbf{z})) '}</BlockMath>
      <BlockMath>{'\\mathcal{L}_{\\text{tight}}(\\mathbf{z}; \\theta, \\hat{\\theta}) = - D(f_\\theta(f_{\\hat{\\theta}}(\\mathbf{z})), f_{\\hat{\\theta}}(\\mathbf{z})) '}</BlockMath>
      <Text mb={5}>where <InlineMath math='D'/> is any distance metric like L1, L2, etc... and <b>we really only optimize <InlineMath math='\theta'/></b>, whereas for <InlineMath math='\hat{\theta}'/> we use a copy of the model that we do not optimize.
      Ultimately, the total loss function is the sum of these terms weighted by some hyper-parameters:</Text>

      <BlockMath>{'\\mathcal{L}(\\theta, \\hat{\\theta}) = \\mathcal{L}_{\\text{rec}}(\\theta) + \\lambda_i \\mathcal{L}_{\\text{idem}}(\\theta, \\hat{\\theta}) + \\lambda_t \\mathcal{L}_{\\text{tight}}(\\theta, \\hat{\\theta})'}</BlockMath>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Implementation</Text>
      <Text mb={5}>Given the simple yet elegant idea, and the fact that this could be tried quite quickly, I could not resist re-implementing this paper. <Link textColor={'blue.500'} href='https://github.com/brianpulfer/idempotent-generative-network'>My full re-implementation</Link> is available on GitHub.</Text>

      <Text mb={5}>The real highlight is the <Code>IdempotentNetwork</Code> lightning module, which can be used to train any model architecture using the above defined objectives:</Text>
      <CodeBlock language={'python'}>
{`from copy import deepcopy

from torch.optim import Adam
from torch.nn import L1Loss
import pytorch_lightning as pl


class IdempotentNetwork(pl.LightningModule):
    def __init__(
        self,
        prior,
        model,
        lr=1e-4,
        criterion=L1Loss(),
        lrec_w=20.0,
        lidem_w=20.0,
        ltight_w=2.5,
    ):
        super(IdempotentNetwork, self).__init__()
        self.prior = prior
        self.model = model
        self.model_copy = deepcopy(model)
        self.lr = lr
        self.criterion = criterion
        self.lrec_w = lrec_w
        self.lidem_w = lidem_w
        self.ltight_w = ltight_w

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return optim

    def get_losses(self, x):
        # Prior samples
        z = self.prior.sample_n(x.shape[0]).to(x.device)

        # Updating the copy
        self.model_copy.load_state_dict(self.model.state_dict())

        # Forward passes
        fx = self(x)
        fz = self(z)
        fzd = fz.detach()

        l_rec = self.lrec_w * self.criterion(fx, x)
        l_idem = self.lidem_w * self.criterion(self.model_copy(fz), fz)
        l_tight = -self.ltight_w * self.criterion(self(fzd), fzd)

        return l_rec, l_idem, l_tight

    def training_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "train/loss_rec": l_rec,
                "train/loss_idem": l_idem,
                "train/loss_tight": l_tight,
                "train/loss": l_rec + l_idem + l_tight,
            },
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "val/loss_rec": l_rec,
                "val/loss_idem": l_idem,
                "val/loss_tight": l_tight,
                "val/loss": loss,
            },
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "test/loss_rec": l_rec,
                "test/loss_idem": l_idem,
                "test/loss_tight": l_tight,
                "test/loss": loss,
            },
            sync_dist=True,
        )

    def generate_n(self, n, device=None):
        z = self.prior.sample_n(n)

        if device is not None:
            z = z.to(device)

        return self(z)
`}
            </CodeBlock>

      <Text mb={5}>The model does just what we covered above: with <Code>l_rec</Code> we encourage the function to act as the identity for in-distribution samples,
      with <Code>l_idem</Code> we encourage the model to output something that will remain the same whether it is mapped through a copy of the model again or not and, finally,
      with <Code>l_tight</Code> we encourage the model to make input and output as different as possible (for when the model acts as a copy trying to disrupt its own output).
      This whole idea is quite unique and brilliant to be fair.</Text>

      <Text mb={5}>Now that we can train any model with the IGN objectives, we just need to write a classical training boilerplate code to try and generate MNIST digits. I went for my favourite stack: Pytorch Lightning with Weights and Biases:</Text>

<CodeBlock language={'python'}>{`import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Lambda
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import DCGANLikeModel
from ign import IdempotentNetwork


def main(args):
    # Set seed
    pl.seed_everything(args["seed"])

    # Load datas
    normalize = Lambda(lambda x: (x - 0.5) * 2)
    noise = Lambda(lambda x: (x + torch.randn_like(x) * 0.15).clamp(-1, 1))
    train_transform = Compose([ToTensor(), normalize, noise])
    val_transform = Compose([ToTensor(), normalize])

    train_set = MNIST(
        root="mnist", train=True, download=True, transform=train_transform
    )
    val_set = MNIST(root="mnist", train=False, download=True, transform=val_transform)

    def collate_fn(samples):
        return torch.stack([sample[0] for sample in samples])

    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )

    # Initialize model
    prior = torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28))
    net = DCGANLikeModel()
    model = IdempotentNetwork(prior, net, args["lr"])

    if not args["skip_train"]:
        # Train model
        logger = WandbLogger(name="IGN", project="Papers Re-implementations")
        callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                dirpath="checkpoints",
                filename="best",
            )
        ]
        trainer = pl.Trainer(
            strategy="ddp",
            accelerator="auto",
            max_epochs=args["epochs"],
            logger=logger,
            callbacks=callbacks,
        )
        trainer.fit(model, train_loader, val_loader)

    # Loading the best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        IdempotentNetwork.load_from_checkpoint(
            "checkpoints/best.ckpt", prior=prior, model=net
        )
        .eval()
        .to(device)
    )

    # Generating images with the trained model
    os.makedirs("generated", exist_ok=True)

    images = model.generate_n(100, device=device)
    save_image(images, "generated.png", nrow=10, normalize=True)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_train", action="store_true")
    args = vars(parser.parse_args())

    main(args)`}
</CodeBlock>

      <Text mb={5}>At first, I found training of IGNs to be unstable. I suspected this might be the case, since we basically have an instance of adversarial training.
      With adversarial training, like in GANs, you might face the problem where the two actors keep on changing, entering a loop where neither converges because each one has to adapt to the moving counterpart.</Text>

      <Text mb={5}>However, I was skeptical of the model used in this work, <b>DCGAN</b> from <i><Link href='https://arxiv.org/abs/1511.06434' textColor={'blue.500'}>Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</Link></i> by <i>Radford et. al.</i>,
      which being from 2015 is stone-age old and does not include modern architectural choices. For example, it completely lacks dropout and normalization layers, as well as residual connections, which are key to any model nowadays.
      While no big models are needed to generate MNIST digits, doing without these core components felt wrong.
      I suspected that this might further harm training stability, so I opted to add batch normalizations and dropout layers here and there.</Text>

<CodeBlock language={'python'}>{`"""DCGAN code from https://github.com/kpandey008/dcgan"""
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_c=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input Size: 1 x 28 x 28
            nn.Conv2d(in_channels, base_c, 4, 2, 1, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input Size: 32 x 14 x 14
            nn.BatchNorm2d(base_c),
            nn.Conv2d(base_c, base_c * 2, 4, 2, 1, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input Size: 64 x 7 x 7
            nn.BatchNorm2d(base_c * 2),
            nn.Conv2d(base_c * 2, base_c * 4, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input Size: 128 x 7 x 7
            nn.BatchNorm2d(base_c * 4),
            nn.Conv2d(base_c * 4, base_c * 8, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input Size: 256 x 7 x 7
            nn.Conv2d(base_c * 8, base_c * 8, 3, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input Size: 256 x 7 x 7
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # Input Size: 128 x 7 x 7
            nn.BatchNorm2d(in_channels // 2),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # Input Size: 64 x 7 x 7
            nn.BatchNorm2d(in_channels // 4),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # Input Size: 32 x 14 x 14
            nn.BatchNorm2d(in_channels // 8),
            nn.ConvTranspose2d(
                in_channels // 8, in_channels // 16, 4, 2, 1, bias=False
            ),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # Input Size : 16 x 28 x 28
            nn.ConvTranspose2d(in_channels // 16, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Final Output : 1 x 28 x 28
        )

    def forward(self, input):
        return self.main(input)


class DCGANLikeModel(nn.Module):
    def __init__(self, in_channels=1, base_c=64):
        super(DCGANLikeModel, self).__init__()
        self.discriminator = Discriminator(in_channels=in_channels, base_c=base_c)
        self.generator = Generator(base_c * 8, out_channels=in_channels)

    def forward(self, x):
        return self.generator(self.discriminator(x))
`}</CodeBlock>

      <Text mb={5}>Surprisingly, this fixed the stability issues during training, although I would not be surprised if training IGNs would turn out to be difficult for some datasets.</Text>
      <Text mb={5}>Here I share the Weights and Biases run of the final model that I used to generate MNIST digits.</Text>
      <iframe src="https://wandb.ai/peutlefaire/Papers%20Re-implementations/runs/1y57sweb?workspace=user-peutlefaire" style={{ border: 'none', width: '100%', height: '1024px' }}></iframe>

      <Text mt={5} mb={5}>And of course, ✨ <i>dulcis in fundo</i> ✨, here are the generated images:</Text>
      <Center mb={5} className="flex flex-col">
        <Image className="flex flex-row" src="/imgs/blog/ign/generated.png" alt="Idempotent Generative Network" />
        <Text textAlign={'center'} textColor={'gray.500'} fontStyle={'italic'} fontSize={'sm'}>
            Generated images with Idempotent Generative Network
        </Text>
      </Center>

      <Text mt={5} mb={10}><b>Thank you</b> for reading! If you found this helpful / interesting, or have suggestions on how to improve, please do not hesitate to contact me at <Link textColor={'blue.500'} href='mailto:me@brianpulfer.ch' color={'blue.500'}>me@brianpulfer.ch</Link></Text>
  </>)
}

IGN.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
        <BlogLayout>
            {page}
        </BlogLayout>
    </AppLayout>
  )
}
