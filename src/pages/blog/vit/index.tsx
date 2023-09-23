import AppLayout from "@/components/Layout/AppLayout"
import BlogLayout from "../layout"
import CodeBlock from "@/components/Blog/CodeBlock"
import { Center, Code, Image, Link, Text} from "@chakra-ui/react"

export default function ViT(){
    return (
        <>
            <Text fontSize={"5xl"} textAlign={"center"}>Vision Transformers from Scratch (PyTorch): A step-by-step guide</Text>



            <Text fontSize={"3xl"} fontWeight={"bold"} mb={5}>Introduction</Text>
            <Text mb={5}>Vision Transformers (ViT), since their introduction by <Link textColor={"blue.500"} href="https://arxiv.org/abs/2010.11929">Dosovitskiy et. al.</Link> in 2020, have dominated the field of Computer Vision, obtaining state-of-the-art performance in image classification first, and later on in other tasks as well.</Text>
            <Text mb={5}>However, unlike other architectures, they are a bit harder to grasp, particularly if you are not already familiar with the Transformer model used in Natural Language Processing (NLP).</Text>
            <Text mb={5}>If you are into Computer Vision (CV) and are still unfamiliar with the ViT model, {"don't"} worry! So was I!</Text>
            <Text mb={5}>In this brief piece of text, I will show you how I implemented my first ViT from scratch (using PyTorch), and I will guide you through some debugging that will help you better visualize what exactly happens in a ViT.</Text>
            <Text mb={5}>While this article is specific to ViT, the concepts you will find here, such as the Multi-headed Self Attention (MSA) block, are present and currently very relevant in various sub-fields of AI, such as CV, NLP, etc…</Text>
            

            <Text fontSize={"3xl"} fontWeight={"bold"} mb={5}>Defining the task</Text>
            <Text mb={5}>Since the goal is just learning more about the ViT architecture, it is wise to pick an easy and well-known task and dataset. In our case, the task is the image classification for the popular MNIST dataset by the great <Link textColor={"blue.500"} href="https://yann.lecun.com/exdb/mnist/">LeCun et. al.</Link></Text>
            <Text mb={5}>{"If you didn’t already know, MNIST is a dataset of hand-written digits ([0–9]) all contained in 28x28 binary pixels images. The task is referred to as trivial for today's algorithms, so we can expect that a correct implementation will perform well."}</Text>
            <Text mb={5}>{"Let’s start with the imports then:"}</Text>
            <CodeBlock language={"python"}>
{`import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)
`}
            </CodeBlock>
            <Text mb={5}>Let’s create a <b>main function</b> that prepares the MNIST dataset, instantiates a model, and trains it for 5 epochs. After that, the loss and accuracy are measured on the test set.</Text>
            <CodeBlock language={"python"}>
{`def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
`}
            </CodeBlock>
            <Text mb={5}>Now that we have this template, from now on, we can just focus on the model (ViT) that will have to classify the images with shape <Code>(N x 1 x 28 x 28)</Code>.</Text>
            <Text mb={5}>Let’s start by defining an empty <Code>nn.Module</Code>. We will then fill this class step by step.</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self):
        # Super constructor
        super(MyViT, self).__init__()

    def forward(self, images):
        pass
`}
            </CodeBlock>
            <Text fontSize={"3xl"} fontWeight={"bold"} mb={5}>Forward pass</Text>
            <Text mb={5}>As Pytorch, as well as most DL frameworks, provides autograd computations, we are only concerned with implementing the forward pass of the ViT model. Since we have defined the optimizer of the model already, the framework will take care of back-propagating gradients and training the model’s parameters.</Text>
            <Text mb={5}>While implementing a new model, I like to keep a picture of the architecture on some tab. Here’s our reference picture for the ViT from <Link textColor={"blue.500"} href="https://www.researchgate.net/publication/348947034_Vision_Transformers_for_Remote_Sensing_Image_Classification">Bazi et. al (2021)</Link>:</Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/arch.png" alt="ViT architecture"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}>The architecture of the ViT with specific details on the transformer encoder and the MSA block. Keep this picture in mind. Picture from <Link href="https://www.researchgate.net/publication/348947034_Vision_Transformers_for_Remote_Sensing_Image_Classification">Bazi et. al.</Link></Text>
            </Center>
            <Text mb={5}>By the picture, we see that the input image (a) is “cut” into sub-images equally sized.</Text>
            <Text mb={5}>Each such sub-image goes through a linear embedding. From then on, each sub-image is just a one-dimensional vector.</Text>
            <Text mb={5}>A positional embedding is then added to these vectors (tokens). The positional embedding allows the network to know where each sub-image is positioned originally in the image. Without this information, the network would not be able to know where each such image would be placed, leading to potentially wrong predictions!</Text>
            <Text mb={5}>These tokens are then passed, together with a special classification token, to the transformer encoders blocks, were each is composed of : A Layer Normalization (LN), followed by a Multi-head Self Attention (MSA) and a residual connection. Then a second LN, a Multi-Layer Perceptron (MLP), and again a residual connection. These blocks are connected back-to-back.</Text>
            <Text mb={5}>Finally, a classification MLP block is used for the final classification only on the special classification token, which by the end of this process has global information about the picture.</Text>
            <Text mb={5}>Let’s build the ViT in <b>6 main steps</b>.</Text>
            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 1: Patchifying and the linear mapping</Text>
            <Text mb={5}>The transformer encoder was developed with sequence data in mind, such as English sentences. However, an image is not a sequence. It is just, uhm… an image… So how do we “sequencify” an image? We break it into multiple sub-images and map each sub-image to a vector!</Text>
            <Text mb={5}>We do so by simply reshaping our input, which has size <Code>(N, C, H, W)</Code> (in our example <Code>(N, 1, 28, 28)</Code>), to size <Code>(N, #Patches, Patch dimensionality)</Code>, where the dimensionality of a patch is adjusted accordingly.</Text>
            <Text mb={5}>In this example, we break each <Code>(1, 28, 28)</Code> into 7x7 patches (hence, each of size 4x4). That is, we are going to obtain 7x7=49 sub-images out of a single image.</Text>
            <Text mb={5}>Thus, we reshape input <Code>(N, 1, 28, 28)</Code> to <Code>(N, PxP, HxC/P x WxC/P) = (N, 49, 16)</Code></Text>
            <Text mb={5}>Notice that, while each patch is a picture of size 1x4x4, we flatten it to a 16-dimensional vector. Also, in this case, we only had a single color channel. If we had multiple color channels, those would also have been flattened into the vector.</Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/patching.png" alt="ViT patching strategy"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}>Raffiguration of how an image is split into patches. The 1x28x28 image is split into 49 (7x7) patches, each of size 16 (4x4x1)</Text>
            </Center>
            <Text mb={5}>We modify our <Code>MyViT</Code> class to implement the patchifying only. We create a method that does the operation from scratch. Notice that this is an inefficient way to carry out the operation, but the code is intuitive for learning about the core concept.</Text>
            <CodeBlock language={"python"}>
{`def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
`}
            </CodeBlock>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        return patches
`}
            </CodeBlock>
            <Text mb={5}>The class constructor now lets the class know the size of our input images (number of channels, height and width). Note that in this implementation, the n_patches variable is the number of patches that we will find both in width and height (in our case it’s 7 because we break the image into 7x7 patches).</Text>
            <Text mb={5}>We can test the functioning of our class with a simple main program:</Text>
            <CodeBlock language={"python"}>
{`if __name__ == '__main__':
    # Current model
    model = MyViT(
        chw=(1, 28, 28),
        n_patches=7
    )

    x = torch.randn(7, 1, 28, 28) # Dummy images
    print(model(x).shape) # torch.Size([7, 49, 16])
`}
            </CodeBlock>
            <Text mb={5}>Now that we have our flattened patches, we can map each of them through a Linear mapping. While each patch was a 4x4=16 dimensional vector, the linear mapping can map to any arbitrary vector size. Thus, we add a parameter to our class constructor, called <Code>hidden_d</Code> for ‘hidden dimension’.</Text>
            <Text mb={5}>In this example, we will use a hidden dimension of 8, but in principle, any number can be put here. We will thus be mapping each 16-dimensional patch to an 8-dimensional patch.</Text>
            <Text mb={5}>We simply create a <Code>nn.Linear</Code> layer and call it in our forward function.</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        return tokens
`}
            </CodeBlock>
            <Text mb={5}>Notice that we run an <Code>(N, 49, 16)</Code> tensor through a (16, 8) linear mapper (or matrix). The linear operation only happens on the last dimension.</Text>

            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 2: Adding the classification token</Text>
            <Text mb={5}>If you look closely at the architecture picture, you will notice that also a <Code>v_class</Code> token is passed to the Transformer Encoder. What’s this?</Text>
            <Text mb={5}>Simply put, this is a special token that we add to our model that has the role of capturing information about the other tokens. This will happen with the MSA block (later on). When information about all other tokens will be present here, we will be able to classify the image using only this special token. The initial value of the special token (the one fed to the transformer encoder) is a parameter of the model that needs to be learned.</Text>
            <Text mb={5}>This is a cool concept of transformers! If we wanted to do another downstream task, we would just need to add another special token for the other downstream task (for example, classifying a digit as higher than 5 or lower) and a classifier that takes as input this new token. Clever, right?</Text>
            <Text mb={5}>We can now add a parameter to our model and convert our <Code>(N, 49, 8)</Code> tokens tensor to an <Code>(N, 50, 8)</Code> tensor (we add the special token to each sequence).</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        return tokens
`}
            </CodeBlock>
            <Text mb={5}>Notice that the classification token is put as the first token of each sequence. This will be important to keep in mind when we will then retrieve the classification token to feed to the final MLP.</Text>
            
            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 3: Positional encoding</Text>
            <Text mb={5}>As anticipated, positional encoding allows the model to understand where each patch would be placed in the original image. While it is theoretically possible to learn such positional embeddings, previous work by <Link textColor={"blue.500"} href="https://arxiv.org/abs/1706.03762">Vaswani et. al.</Link> suggests that we can just add sines and cosines waves.</Text>
            <Text mb={5}>In particular, positional encoding adds low-frequency values to the first dimensions and higher-frequency values to the latter dimensions.</Text>
            <Text mb={5}>In each sequence, for token i we add to its j-th coordinate the following value:</Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/embedding.png" alt="ViT embedding"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}>Value to be added to the i-th tensor in its j-th coordinate. <Link href="https://blogs.oracle.com/ai-and-datascience/post/multi-head-self-attention-in-nlp">Image source</Link>.</Text>
            </Center>
            <Text mb={5}>This positional embedding is a function of the number of elements in the sequence and the dimensionality of each element. Thus, it is always a 2-dimensional tensor or “rectangle”.</Text>
            <Text mb={5}>Here’s a simple function that, given the number of tokens and the dimensionality of each of them, outputs a matrix where each coordinate <Code>(i,j)</Code> is the value to be added to token <Code>i</Code> in dimension <Code>j</Code>.</Text>
            <CodeBlock language={"python"}>
{`def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.imshow(get_positional_embeddings(100, 300), cmap="hot", interpolation="nearest")
    plt.show()
`}
            </CodeBlock>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/embedding_matrix.png" alt="Embedding matrix"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}>Heatmap of Positional embeddings for one hundred 300-dimensional samples. Samples are on the y-axis, whereas the dimensions are on the x-axis. Darker regions show higher values.</Text>
            </Center>
            <Text mb={5}>From the heatmap we have plotted, we see that all ‘horizontal lines’ are all different from each other, and thus samples can be distinguished.</Text>
            <Text mb={5}>We can now add this positional encoding to our model after the linear mapping and the addition of the class token.</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed
        return out
`}
            </CodeBlock>
            <Text mb={5}>We define the positional embedding to be a parameter of our model (that we won’t update by setting its <Code>requires_grad</Code> to <Code>False</Code>). Note that in the forward method, since tokens have size <Code>(N, 50, 8)</Code>, we have to repeat the <Code>(50, 8)</Code> positional encoding matrix <Code>N</Code> times.</Text>
            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 4: The encoder block (Part 1/2)</Text>
            <Text mb={5}>This is possibly the hardest step of all. An encoder block takes as input our current tensor <Code>(N, S, D)</Code> and outputs a tensor of the same dimensionality.</Text>
            <Text mb={5}>The first part of the encoder block applies Layer Normalization to our tokens, then a Multi-head Self Attention, and finally adds a residual connection.</Text>
            <Text fontSize={"l"} fontWeight={"bold"} mb={5}>Layer Normalization</Text>
            <Text mb={5}>Layer normalization is a popular block that, given an input, subtracts its mean and divides by the standard deviation.</Text>
            <Text mb={5}>However, we commonly apply layer normalization to an <Code>(N, d)</Code> input, where d is the dimensionality. Luckily, also the Layer Normalization module generalizes to multiple dimensions, check this:</Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/layernorm.png" alt="Layer normalization"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}><Code>nn.LayerNorm</Code> can be applied in multiple dimensions. We can normalize fifty 8-dimensional vectors, but we can also normalize sixteen by fifty 8-dimensional vectors.</Text>
            </Center>
            <Text mb={5}>Layer normalization is applied to the last dimension only. We can thus make each of our 50x8 matrices (representing a single sequence) have mean 0 and std 1. After we run our <Code>(N, 50, 8)</Code> tensor through LN, we still get the same dimensionality.</Text>
            <Text fontSize={"l"} fontWeight={"bold"} mb={5}>Multi-Head Self-Attention</Text>
            <Text mb={5}>We now need to implement sub-figure c of the architecture picture. What’s happening there?</Text>
            <Text mb={5}>Simply put: we want, for a single image, each patch to get updated based on some similarity measure with the other patches. We do so by linearly mapping each patch (that is now an 8-dimensional vector in our example) to 3 distinct vectors: <b>q</b>, <b>k</b>, and <b>v</b> (query, key, value).</Text>
            <Text mb={5}>Then, for a single patch, we are going to compute the dot product between its <b>q</b> vector with all of the <b>k</b> vectors, divide by the square root of the dimensionality of these vectors (sqrt(8)), softmax these so-called attention cues, and finally multiply each attention cue with the <b>v</b> vectors associated with the different <b>k</b> vectors and sum all up.</Text>
            <Text mb={5}>In this way, each patch assumes a new value that is based on its similarity (after the linear mapping to <b>q</b>, <b>k</b>, and <b>v</b>) with other patches. This whole procedure, however, is carried out <b>H</b> times on <b>H</b> sub-vectors of our current 8-dimensional patches, where <b>H</b> is the number of Heads. If you’re unfamiliar with the attention and multi-head attention mechanisms, I suggest you read this <Link textColor={"blue.500"} href="https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/">nice post</Link> by <Link textColor={"blue.500"} href="https://data-science-blog.com/blog/author/yasuto/">Yasuto Tamura</Link>.</Text>
            <Text mb={5}>Once all results are obtained, they are concatenated together. Finally, the result is passed through a linear layer (for good measure).</Text>
            <Text mb={5}>The intuitive idea behind attention is that it allows modeling the relationship between the inputs. What makes a ‘0’ a zero are not the individual pixel values, but how they relate to each other.</Text>
            <Text mb={5}>Since quite some computations are carried out, it is worth creating a new class for MSA:</Text>
            <CodeBlock language={"python"}>
{`class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
`}
            </CodeBlock>
            <Text mb={5}>Notice that, for each head, we create distinct Q, K, and V mapping functions (square matrices of size 4x4 in our example).</Text>
            <Text mb={5}>Since our inputs will be sequences of size <Code>(N, 50, 8)</Code>, and we only use 2 heads, we will at some point have an <Code>(N, 50, 2, 4)</Code> tensor, use a <Code>nn.Linear(4, 4)</Code> module on it, and then come back, after concatenation, to an <Code>(N, 50, 8)</Code> tensor.</Text>
            <Text mb={5}>Also notice that using loops is not the most efficient way to compute the multi-head self-attention, but it makes the code much clearer for learning.</Text>
            <Text fontSize={"l"} fontWeight={"bold"} mb={5}>Residual connection</Text>
            <Text mb={5}>A residual connection consists in just adding the original input to the result of some computation. This, intuitively, allows a network to become more powerful while also preserving the set of possible functions that the model can approximate.</Text>
            <Text mb={5}>We will add a residual connection that will add our original <Code>(N, 50, 8)</Code> tensor to the <Code>(N, 50, 8)</Code> obtained after LN and MSA. It’s time to create the transformer encoder block class, which will be a component of the <Code>MyViT</Code> class:</Text>
            <CodeBlock language={"python"}>
{`class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        return out
`}
            </CodeBlock>
            <Text mb={5}>Phew, that was quite some work! But I promise this was the hardest part. From now on, it’s all downhill.</Text>
            <Text mb={5}>With this self-attention mechanism, the class token (first token of each of the N sequences) now has information regarding all other tokens!</Text>
            
            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 5: The encoder block (Part 2/2)</Text>
            <Text mb={5}>All that is left to the transformer encoder is just a simple residual connection between what we already have and what we get after passing the current tensor through another LN and an MLP. The MLP is composed of two layers, where the hidden layer typically is four times as big (this is a parameter)</Text>
            <CodeBlock language={"python"}>
{`class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
`}
            </CodeBlock>
            <Text mb={5}>We can indeed see that the Encoder block outputs a tensor of the same dimensionality:</Text>
            <CodeBlock language={"python"}>
{`if __name__ == '__main__':
    model = MyVitBlock(hidden_d=8, n_heads=2)

    x = torch.randn(7, 50, 8)  # Dummy sequences
    print(model(x).shape)      # torch.Size([7, 50, 8])
`}
            </CodeBlock>
            <Text mb={5}>Now that the encoder block is ready, we just need to insert it in our bigger ViT model which is responsible for patchifying before the transformer blocks, and carrying out the classification after.</Text>
            <Text mb={5}>We could have an arbitrary number of transformer blocks. In this example, to keep it simple, I will use only 2. We also add a parameter to know how many heads does each encoder block will use.</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        return out
`}
            </CodeBlock>
            <Text mb={5}>Once more, if we run a random <Code>(7, 1, 28, 28)</Code> tensor through our model, we still get a <Code>(7, 50, 8)</Code> tensor.</Text>
            
            <Text fontSize={"xl"} fontWeight={"bold"} mb={5}>Step 6: Classification MLP</Text>
            <Text mb={5}>Finally, we can extract just the classification token (first token) out of our N sequences, and use each token to get N classifications.</Text>
            <Text mb={5}>Since we decided that each token is an 8-dimensional vector, and since we have 10 possible digits, we can implement the classification MLP as a simple 8x10 matrix, activated with the SoftMax function.</Text>
            <CodeBlock language={"python"}>
{`class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution
`}
            </CodeBlock>
            <Text mb={5}>The output of our model is now an <Code>(N, 10)</Code> tensor. Hurray, we are done!</Text>

            <Text fontSize={"3xl"} fontWeight={"bold"} mb={5}>Results</Text>
            <Text mb={5}>We change the only line in the main program that was previously undefined.</Text>
            <CodeBlock language={"python"}>
{`model = MyVit((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
`}
            </CodeBlock>
            <Text mb={5}>We now just need to run the training and test loops and see how our model performs. If you’ve set torch seed manually (to 0), you should get this printed:</Text>
            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/vit/results.png" alt="ViT results"/>
                <Text textColor={"gray.500"} fontSize={"sm"} textAlign={"center"}>Training losses, test loss, and test accuracy obtained.</Text>
            </Center>
            <Text mb={5}>And that’s it! We have now created a ViT from scratch. Our model achieves <b>~80% accuracy in just 5 epochs</b> and with few parameters.</Text>
            <Text mb={5}>You can find the full script at the following <Link href="https://github.com/BrianPulfer/PapersReimplementations/blob/main/vit/vit_torch.py" textColor={"blue.500"}>link</Link>. Let me know if this post was useful or think something was unclear!</Text>

        </>
    )
}

ViT.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            <BlogLayout>
                {page}
            </BlogLayout>
        </AppLayout>
    )
}