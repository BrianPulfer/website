import AppLayout from '@/components/Layout/AppLayout'
import BlogLayout from '../layout'
import Head from 'next/head'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'
import CodeBlock from '@/components/Blog/CodeBlock'
import { Link, Text } from '@chakra-ui/react'

export default function ViR (): JSX.Element {
  return (
    <>
      <Head><title>Blog - ViR</title></Head>
      <Text fontSize={'l'} textAlign={'right'}><b>Published:</b> 06.12.2023</Text>
      <Text fontSize={'5xl'} textAlign={'center'}>Vision Retention Networks</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Introduction</Text>
      <Text mb={5}>Retention is a mechanism recently proposed in <Link textColor={'blue.500'} href='https://arxiv.org/abs/2307.08621'>Retentive Network: A Successor to Transformer for Large Language Models</Link> by <i>Sun et. al.</i> which core idea is to carry out similar computation as attention while being much more computationally efficient.
                It has now become a recurrent pattern that researchers from other fields of machine learning take inspiration from the progress done in NLP and try to adapt NLP-solutions to a different problem.
                It was thus only a matter of time before we would have heard of Retention in the field of Computer Vision.</Text>
      <Text mb={5}><Link textColor={'blue.500'} href='https://arxiv.org/abs/2310.19731'>ViR: Vision Retention Networks</Link> by <i>Ali Hatamizadeh, Michael Ranzinger, Jan Kautz</i> first applied Retention in a CV model.
                I recently had a great time re-implementing the paper and digging into Retention, so I thought I would share what I have learned.
                You can find my re-implementation at <Link textColor={'blue.500'} href={'https://github.com/brianpulfer/vision-Retention-networks'}>brianpulfer/vision-Retention-networks</Link> or at <Link textColor={'blue.500'} href='https://github.com/brianpulfer/papersreimplementations'>brianpulfer/papersreimplementations</Link>.</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Attention</Text>
      <Text mb={5}>Before we dig into ViR, we need to learn what Retention is. But before we learn what Retention is, a little recap on Attention.</Text>
      <Text mb={5}>Attention is a mechanism that allows a model to learn relationships between elements of the input. The meaning of a word can be completely altered based on the surrounding words. A red pixel in an image might come from a tomato, a Ferrari car or a baloon. Only the combination with neighbouring pixels give it a meaning.
                It is thus important for models to have the ability to learn the interplay of elements in the input sequence. That is where Attention comes in, and this is how it works:</Text>

      <Text mb={5}>
        Given an input sequence <InlineMath math='X \in \mathbb{R} ^ {N \times D}' />, attention computes Queries, Keys and Values for each element of the sequence as follows:
      </Text>
      <BlockMath>{'Q = X W_q'}</BlockMath>
      <BlockMath>{'K = X W_k'}</BlockMath>
      <BlockMath>{'V = X W_v'}</BlockMath>
      <Text mb={5}>
        Where <InlineMath math='W_q, W_k, W_v \in \mathbb{R} ^ {D \times D}' /> are learnable parameters. The output for each element of the sequence is going to be a weighted sum of the values, where the weights are computed as the dot product between the query and the keys:
      </Text>
      <BlockMath>{'\\text{Attention}(Q, K, V) = \\text{softmax} \\left( \\frac{Q K^T}{\\sqrt{D}} \\right) V'}</BlockMath>
      <Text mb={5}>and <InlineMath math='\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}' /> is applied row-wise.</Text>
      <Text mb={5}>This mechanism, ever since the <Link textColor={'blue.500'} href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</Link> paper, has been empirically proven to be very powerful for learning relationships between elements of a sequence. It has been used in virtually all contexts (NLP, CV, TTS, ...), and it has become a de-facto standard for many tasks.</Text>

      <Text textAlign={'center'} mb={5}><i>Then why getting rid of it?</i></Text>
      <Text mb={5}>There is only one issue that has researchers a bit troubled: the complexity of attention is <InlineMath math='O(N^2)' /> (easily seen when computing <InlineMath math='QK^T' />), meaning that for an input sequence twice as long, computing attention takes four times as much time.</Text>
      <Text mb={5}>Quite some effort went into trying to solve this issue, with various variations like <Link textColor={'blue.500'} href='https://arxiv.org/abs/2007.14902'>Linear Attention</Link> and <Link textColor={'blue.500'} href='https://arxiv.org/abs/1812.01243'>Efficient Attention</Link> trying to replicate the mechanism while being computationally more convenient.</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Retention</Text>
      <Text mb={5}>Retention works recurrently just like recurrent neural networks. At each step, it reads the input to update an inner state matrix, use the inner state to compute an output and pass the inner state onward. Here is the <b style={{ color: 'orange' }}>RECURRENT</b> formulation of Retention</Text>
      <BlockMath>{'\\mathbf{s}_n = \\alpha \\mathbf{s}_{n-1} + \\mathbf{k}_n^t \\mathbf{v}_n'}</BlockMath>
      <BlockMath>{'\\text{Retention}(\\mathbf{x})=\\mathbf{o}_n = \\mathbf{q}_n \\mathbf{s}_n'}</BlockMath>

      <Text mb={5}>where <InlineMath math='\mathbf{s}_n' /> is the inner state at step <InlineMath math='n' />, <InlineMath math='\mathbf{k}_n' /> is the key and <InlineMath math='\mathbf{v}_n' /> is the value of the current (n-th) element in the sequence (row vectors, so <InlineMath math='\mathbf{s}_n, \mathbf{k_n^Tv_n} \in \mathbb{R}^{D \times D}' />).
        Needless to say, <InlineMath math='\mathbf{q}_n, \mathbf{k}_n, \mathbf{v}_n' /> are linear projections of the n-th sequence element <InlineMath math='\mathbf{x}_n' />. Finally, <InlineMath math='0 \le \alpha \le 1' /> is a constant that exponentially decays older key-values products.</Text>

      <Text mb={5}>Translating into text these equations, the idea is the following: <InlineMath math='\mathbf{s}_n' /> will contain the <i>state</i> in the form of all key-value products. The ouput is obtained by fetching the desired value (mixture of values) by using the current query <InlineMath math='\mathbf{q}_n'></InlineMath>.</Text>

      <Text>This is literally all there is to Retention! What is so special about it is that it can also be computed using a <b style={{ color: 'orange' }}>PARALLEL</b> formulation just like we do for Attention. The formula to compute all outputs at once is the following:</Text>
      <BlockMath>{'M_{i,j} = \\begin{cases} 0, & i < j  \\\\ \\alpha^{i-j}, & i \\ge j \\end{cases}'}</BlockMath>
      <BlockMath>{'\\text{Retention}(X) = (\\frac{QK^T}{\\sqrt{D}} \\odot M)V'}</BlockMath>
      <Text mb={5}>Looks familiar, right? In fact, we do everything exactly as for Attention, except that we do not apply a row-wise softmax function and always apply <InlineMath math='M' />, a lower-triangular matrix that simultaneously deals with causal masking (take no contribution from future elements in the sequence) and applies the exponential decay given by <InlineMath math='\alpha' />.</Text>
      <Text mb={5}>The key takeaway here is that if we get rid of the softmax operator <i>we unlock the recurrent formulation</i>, where we can just carry on what we had computed before to compute the next output.</Text>

      <Text mb={5}>However, processing sequences recurrently sucks! That is exactly the reason why we generally prefer Transformers over RNNs: the Transformer time complexity might be quadratic in sequence length, but at least we can process everything in parallel.
        With a recurrent formulation, we need to sequentially compute the n-th output before we can compute the n-th + 1 while our GPUs sit quiet.</Text>

      <Text textAlign={'center'} mb={5}><i>Then why caring about a recurrent formulation?</i></Text>
      <Text mb={5}>The real ✨magic✨ happens when we decide to use a hybrid between parallel and recurrent formulations. In fact, it turns out that we can break the input sequence into multiple chunks, run each chunk in parallel using the parallel formulation, and then aggregate all of the results with a cross-chunk recurrent computation.
        This means that as soon as the sequence becomes prohibitively long for the parallel formulation (quadratic in <InlineMath math='N' />), we can just split it into chunks of size <InlineMath math='C' /> and run those parallelly (quadratic in chunk-size <InlineMath math='C' /> only!) and finally combine the cross-chunk information recurrently (linear in <InlineMath math='\frac{N}{C}' />).
        The real gain is thus obtained when we have very long sequences.</Text>
      <Text mb={5}>Here we have the <b style={{ color: 'orange' }}>CHUNKWISE RECURRENT</b> formulation of Retention:</Text>
      <BlockMath>{'Q_{[i]} = Q_{Ci:C(i+1)}, \\quad K_{[i]} = K_{Ci:C(i+1)}, \\quad V_{[i]} = V_{Ci:C(i+1)}'}</BlockMath>
      <BlockMath>{'R_i = K^T_{[i]}(V_{[i]} \\odot \\Alpha) + \\alpha^CR_{i-1}, \\quad \\Alpha_{ij} = \\alpha^{C-i-1} '}</BlockMath>
      <BlockMath>{'\\text{Retention}(X_{i}) = \\underbrace{(Q_{[i]} K^T_{[i]} \\odot M)V_{[i]}}_{\\text{Inner-Chunk}} + \\underbrace{(Q_{[i]}R_{i-1}) \\odot \\xi}_{\\text{Intra-Chunk}}, \\quad \\xi_{ij} = \\alpha^{i+1}'}</BlockMath>
      <Text mb={5}>The math looks scary, but really we are just applying the parallel computation for all chunks and, once we have the <i>Inner-Chunk</i> parts, we can merge them using the recurrent formulation.</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Comparison of Attention and Retention</Text>
      <Text mb={5} fontSize={'xl'}>Time complexity</Text>
      <Text mb={5}>Retention can, given the previous state, compute the next token in <InlineMath math='O(1)' /> time complexity, whereas Attention does not have a previous state and it needs to use all <InlineMath math='O(N)' /> past keys and queries to predict the next token.</Text>

      <Text mb={5} fontSize={'xl'}>Recurrent formulation</Text>
      <Text mb={5}>Attention does not need to be formulated recurrently, whereas Retention does. This is perfectly fine for causal decoder transformers, where we don not want current tokens to attend to future tokens anyways.
        However, in computer vision we mostly use the encoder type of transformer, so it is not completely clear what impact forcing the causal relationship might have in a task where seemingly there is no causal relationship.</Text>

      <Text mb={5}><i><b>Personal observation:</b> Because Retention accumulates all keys and queries, I believe that it is probably not as powerful of a mechanism as Attention.
        Perhaps this loss of expressivity is not a big deal for text and/or images, especially compared to the gains made in time complexity, but this is still something to keep in mind.
        It might very well be that Retention fails to become a de-facto standard like other alternatives to Attention before it due to worse performances.
        What is sure is that Retention enables faster inference and, for very long sequences, even faster training while being quite similar to Attention.</i></Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Vision Retention Networks</Text>
      <Text mb={5}>Vision Retention Networks are a minor yet important variantion from Vision Transformer.
        I have previously written about <Link textColor={'blue.500'} href='/blog/vit'>how Vision Transformers (ViT) work</Link>, but in short, a ViT breaks an image into many distinct non-overlapping patches (typically, 16x16 patches of size 14x14 for images of size 224x224) which are then flattened and treated as a sequence. An encoder transformer is then used to process the sequence without any causal masking and the output is used for down-stream tasks.</Text>
      <Text mb={5}>The ViT is thus just a stack of encoder blocks, where each block sequentially applies an <b>Attention</b> block and an <b>MLP</b> block. In ViR, we get rid of the Attention block and swap a Retention block in instead.</Text>
      <Text mb={5}><i><b>Personal observation:</b> It must be noted that because Retention works in a recurrent matter by definition, this is a big shift from ViT! While a ViT sees the whole image in one go, a ViR virtually reads the image from left to right from top to bottom. This is potentially a drawback of ViR over ViT, since it might not make sense to introduce causality in images.</i></Text>
      <Text mb={5}>Because retention reads the image in sequence, if we want our model to be a classifier, we need to use an output that comes after all tokens have been seen.
        To do so, we append a learnable <b>[CLS] at the end</b> of the sequence and use the generated output to do classification.
        Notice than in regular ViT, the CLS token was typically placed at the beginning of the sequence (although for a regular ViT this does not really make a difference).</Text>

      <Text fontSize={'3xl'} fontWeight={'bold'} mb={5}>Implementation</Text>
      <Text mb={5}>Here is <Link color={'blue.500'} href='https://github.com/brianpulfer/vision-retention-networks'>my full re-implementation</Link> of a ViR:</Text>

      <CodeBlock language={'python'}>{
`import torch
import torch.nn as nn


class ViRModes:
    PARALLEL = "parallel"
    RECURRENT = "recurrent"
    CHUNKWISE = "chunkwise"


class Retention(nn.Module):
    def __init__(
        self,
        embed_dim,
        max_len,
        alpha,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
    ):
        super(Retention, self).__init__()
        self.dim = embed_dim
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.mode = mode

        # Useful buffers
        self.register_buffer("dim_sqrt", torch.tensor(embed_dim**0.5))

        indices = torch.arange(max_len).reshape(1, -1)
        self.register_buffer(
            "decay_mask",
            (alpha ** (indices.t() - indices)).tril(),
        )

        self.register_buffer("causal_mask", torch.ones(max_len, max_len).tril())
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

    def forward_parallel(self, x):
        # Getting queries, keys, values
        bs, sl, d = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Causal and decay masking
        M = (self.causal_mask[:sl, :sl] * self.decay_mask[:sl, :sl]).repeat(bs, 1, 1)

        # Retention
        out = (q @ k.transpose(-1, -2) / self.dim_sqrt * M) @ v

        return out

    def forward_recurrent(self, x, state):
        batch_size, length, dim = x.shape

        all_outputs = []
        state = torch.zeros(batch_size, dim, dim, device=x.device)
        for i in range(length):
            xi = x[:, i]
            q, k, v = self.qkv(xi).chunk(3, dim=-1)

            state = self.alpha * state + k.unsqueeze(-1) @ v.unsqueeze(1)
            out = q.unsqueeze(1) @ state / self.dim_sqrt
            all_outputs.append(out.squeeze())

        x = torch.stack(all_outputs, dim=1)
        return x

    def forward_chunkwise(self, x, chunk_size=None):
        # Getting queries, keys, values
        if chunk_size is None:
            chunk_size = self.chunk_size

        bs, sl, d = x.shape

        # Adding dummy tokens to make the sequence length divisible by chunk_size
        if sl % chunk_size != 0:
            x = torch.cat(
                [x, torch.zeros(bs, chunk_size - sl % chunk_size, d, device=x.device)],
                dim=1,
            )
        n_chunks = x.shape[1] // chunk_size

        # Running all chunks in parallel
        x = x.reshape(bs, n_chunks, chunk_size, d)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        M = (
            self.causal_mask[:chunk_size, :chunk_size]
            * self.decay_mask[:chunk_size, :chunk_size]
        ).repeat(bs, n_chunks, 1, 1)

        inner_chunk = (q @ k.transpose(-1, -2) / self.dim_sqrt * M) @ v

        # Updating outputs with chunk-wise recurrent
        retention_mask = (
            torch.tensor(
                [self.alpha ** (chunk_size - i - 1) for i in range(chunk_size)],
                device=x.device,
            )
            .repeat(bs, d, 1)
            .transpose(-1, -2)
        )

        cross_mask = (
            torch.tensor(
                [self.alpha ** (i + 1) for i in range(chunk_size)], device=x.device
            )
            .repeat(bs, n_chunks, d, 1)
            .transpose(-1, -2)
        )

        states = torch.zeros(bs, n_chunks, d, d, device=x.device)
        for i in range(1, n_chunks):
            chunk_state = k[:, i - 1].transpose(-1, -2) @ (v[:, i - 1] * retention_mask)
            states[:, i] = chunk_state + states[:, i - 1] * self.alpha**chunk_size

        cross_chunk = (q @ states) / self.dim_sqrt * cross_mask

        # Combining inner and cross chunk
        out = inner_chunk + cross_chunk

        # Removing dummy tokens
        out = out.flatten(1, 2)[:, :sl]
        return out

    def forward(self, x, state=None, mode=ViRModes.PARALLEL, chunk_size=None):
        if mode is None:
            mode = self.mode

        if mode == ViRModes.PARALLEL:
            return self.forward_parallel(x)
        elif mode == ViRModes.RECURRENT:
            return self.forward_recurrent(x, state)
        elif mode == ViRModes.CHUNKWISE:
            return self.forward_chunkwise(x, chunk_size)
        else:
            raise ValueError(f"Unknown mode {mode}")


class MultiHeadRetention(nn.Module):
    def __init__(
        self,
        heads,
        embed_dim,
        max_len,
        alphas=None,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
    ):
        super(MultiHeadRetention, self).__init__()
        self.n_heads = heads
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.alphas = alphas
        self.head_dim = embed_dim // heads
        self.mode = mode
        self.chunk_size = chunk_size

        if alphas is None:
            alphas = [1 - 2 ** (-5 - i) for i in range(heads)]

        assert len(alphas) == heads, "Number of alphas must match number of heads"

        assert (
            embed_dim % heads == 0
        ), "Embedding dimension must be divisible by the number of heads"

        self.heads = nn.ModuleList(
            [
                Retention(embed_dim // heads, max_len, alpha, mode, chunk_size)
                for alpha in alphas
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mode=None, chunk_size=None):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        out = torch.cat(
            [
                head(
                    x[:, :, i * self.head_dim : (i + 1) * self.head_dim],
                    mode=mode,
                    chunk_size=chunk_size,
                )
                for i, head in enumerate(self.heads)
            ],
            dim=-1,
        )
        return self.linear(self.gelu(self.ln(out)))


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class ViRBlock(nn.Module):
    def __init__(
        self,
        heads,
        embed_dim,
        max_len,
        alphas=None,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
        dropout=0.1,
    ):
        super(ViRBlock, self).__init__()
        self.mode = mode
        self.chunk_size = chunk_size

        self.ln1 = nn.LayerNorm(embed_dim)
        self.retention = MultiHeadRetention(
            heads, embed_dim, max_len, alphas, mode, chunk_size
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mode=None, chunk_size=None):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        x = (
            self.dropout1(self.retention(self.ln1(x), mode=mode, chunk_size=chunk_size))
            + x
        )
        x = self.dropout2(self.mlp(self.ln2(x))) + x
        return x


class ViR(nn.Module):
    def __init__(
        self,
        patch_size=14,
        depth=12,
        heads=12,
        embed_dim=768,
        max_len=256,
        alphas=None,
        mode=ViRModes.CHUNKWISE,
        chunk_size=256,
        dropout=0.1,
    ):
        super(ViR, self).__init__()

        # Local parameters
        self.out_dim = 10
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.alphas = alphas
        self.mode = mode
        self.chunk_size = chunk_size

        # Embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

        # ViR blocks
        self.blocks = nn.ModuleList(
            [
                ViRBlock(heads, embed_dim, max_len, alphas, mode, chunk_size, dropout)
                for _ in range(depth)
            ]
        )

        # Head
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, mode=None, chunk_size=None, reshape=False):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        # Patch embedding, positional embedding
        x = self.patch_embed(x).permute(0, 2, 3, 1).flatten(1, 2)
        bs, sl = x.shape[:2]
        x = x + self.pos_embed.repeat(bs, 1, 1)[:, :sl]

        # Blocks
        for block in self.blocks:
            x = block(x, mode=mode, chunk_size=chunk_size)

        # Layer Norm
        x = self.ln(x)

        # Reshape
        if reshape:
            ps = int(x.shape[1] ** 0.5)
            x = x.reshape(bs, ps, ps, self.embed_dim).permute(0, 3, 1, 2)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(16, 3, 224, 224).to(device)
    model = ViR(depth=12, heads=3, embed_dim=192).eval().to(device)

    with torch.no_grad():
        y1 = model(x, mode=ViRModes.PARALLEL)
        y2 = model(x, mode=ViRModes.RECURRENT)
        y3 = model(x, mode=ViRModes.CHUNKWISE, chunk_size=20)

        assert torch.allclose(
            y1, y2, atol=1e-5
        ), "Parallel and recurrent modes should give the same output"

        assert torch.allclose(
            y1, y3, atol=1e-5
        ), "Parallel and chunkwise modes should give the same output"
`
}</CodeBlock>

      <Text mb={5}>It feels like I should comment there 300+ lines, but really there is nothing that is not already covered in the formulas.
                The only thing that I should mention is that the chunk size <InlineMath math='C' /> might not entirely devide the sequence length <InlineMath math='N' />, so what one can do is adding some dummy tokens at the end of the sequence such that the sequence is entirely divisible by the chunk size (a sort of padding).</Text>

      <Text mb={5}>Also, I found it key for performances to actually perform computations for all chunks in parallel, so it is not enough to re-use the <i>forward_parallel</i> function sequentially for each chunk.</Text>
      <Text mb={5}>Also notice that we use different alphas for each head: some heads with a higher alpha will look further back into the past, other heads with a lower alpha will mostly focus on most recent tokens.</Text>

    <Text mt={5} mb={10}><b>Thank you</b> for reading until here! If you found this helpful / interesting, or have suggestions on how to improve, please do not hesitate to contact me at <Link href='mailto:me@brianpulfer.ch' color={'blue.500'}>me@brianpulfer.ch</Link></Text>
  </>)
}

ViR.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
        <BlogLayout>
            {page}
        </BlogLayout>
    </AppLayout>
  )
}
