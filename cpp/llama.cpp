#include <iostream>
#include <cmath>

#include <torch/torch.h>
#include <sentencepiece_processor.h>

int sample(const torch::Tensor& logits, double temperature) {
  if (temperature < 1e-6) {
    return logits.argmax().item<int>();
  } else {
    torch::Tensor probs = torch::softmax(logits / temperature, -1);
    return torch::multinomial(probs, 1).item<int>();
  }
}

torch::Tensor precompute_freqs_cis(int dim, int end, float theta = 10000.0) {
  torch::Tensor indices = torch::arange(0, dim, 2);
  indices.slice(0, dim/2).to(torch::kFloat);
  torch::Tensor freq_denominator = torch::pow(theta, indices / dim);
  torch::Tensor freqs = 1.0 / freq_denominator;
  torch::Tensor t = torch::arange(end, torch::TensorOptions().device(freqs.device())); // type: ignore
  freqs = torch::outer(t, freqs).to(torch::kFloat32); // type: ignore
  torch::Tensor freqs_cis = torch::polar(torch::ones_like(freqs), freqs);
  return freqs_cis;
}

torch::Tensor reshape_for_broadcast(const torch::Tensor& freqs_cis, const torch::Tensor& x) {
    int ndim = x.dim();
    assert(0 <= 1 && 1 < ndim);
    assert(freqs_cis.sizes() == torch::IntArrayRef({x.size(1), x.size(-1)}));

    std::vector<int64_t> shape;
    for (int i = 0; i < ndim; ++i) {
        shape.push_back((i == 1 || i == ndim - 1) ? x.size(i) : 1);
    }

    return freqs_cis.view(shape);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb(torch::Tensor xq, torch::Tensor xk, torch::Tensor freqs_cis) {
    // Convert xq and xk to complex numbers
    torch::Tensor xq_ = torch::view_as_complex(xq.to(torch::kFloat32).reshape({xq.size(0), xq.size(1), xq.size(2), -1, 2}));
    torch::Tensor xk_ = torch::view_as_complex(xk.to(torch::kFloat32).reshape({xk.size(0), xk.size(1), xk.size(2), -1, 2}));

    // Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_);

    // Apply rotary embeddings
    torch::Tensor xq_out = torch::view_as_real(xq_ * freqs_cis).flatten(3);
    torch::Tensor xk_out = torch::view_as_real(xk_ * freqs_cis).flatten(3);

    // Convert back to the original data type and return
    return std::make_tuple(xq_out.to(xq.scalar_type()), xk_out.to(xk.scalar_type()));
}

class RMSNormImpl : public torch::nn::Module {
public:
  RMSNormImpl(int64_t dim, double eps = 1e-6)
    : eps(eps),
      weight(torch::ones({dim}, torch::TensorOptions().device(torch::kCUDA))),
      dim(dim) {
    register_parameter("weight", weight);
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor output = _norm(x.to(torch::kFloat32)).to(x.scalar_type());
    return output * weight;
  }
private:
  double eps;
  torch::Tensor weight;
  int64_t dim;
  torch::Tensor _norm(torch::Tensor x) {
    return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps);
  }
};
TORCH_MODULE(RMSNorm);

class AttentionImpl : public torch::nn::Module {
public:
  AttentionImpl(int64_t dim, int64_t n_heads)
    : wq(torch::nn::Linear(dim, dim)),
      wk(torch::nn::Linear(dim, dim)),
      wv(torch::nn::Linear(dim, dim)),
      wo(torch::nn::Linear(dim, dim)),
      n_heads(n_heads),
      head_dim(dim / n_heads) {
    register_module("wq", wq);
    register_module("wk", wk);
    register_module("wv", wv);
    register_module("wo", wo);
  }

  torch::Tensor forward(torch::Tensor x, int64_t start_pos, torch::Tensor freqs_cis, torch::Tensor mask) {
    int64_t bsz = x.size(0);
    int64_t seqlen = x.size(1);

    torch::Tensor xq = wq(x);
    torch::Tensor xk = wk(x);
    torch::Tensor xv = wv(x);

    xq = xq.view({bsz, seqlen, n_heads, head_dim});
    xk = xk.view({bsz, seqlen, n_heads, head_dim});
    xv = xv.view({bsz, seqlen, n_heads, head_dim});

    auto result = apply_rotary_emb(xq, xk, freqs_cis);
    xq = std::get<0>(result);
    xk = std::get<1>(result);

    xq = xq.transpose(1, 2);
    torch::Tensor keys = xk.transpose(1, 2);
    torch::Tensor values = xv.transpose(1, 2);

    torch::Tensor scores = torch::matmul(xq, keys.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));

    if (mask.defined()) {
      scores += mask;
    }

    scores = torch::softmax(scores, -1);
    torch::Tensor output = torch::matmul(scores, values).transpose(1, 2).contiguous().view({bsz, seqlen, -1});
    return wo(output);
  }
private:
  torch::nn::Linear wq, wk, wv, wo;
  int64_t dim, n_heads, head_dim;
};
TORCH_MODULE(Attention);

class FeedForwardImpl : public torch::nn::Module {
public:
  FeedForwardImpl(int64_t dim, int64_t hidden_dim, int64_t multiple_of)
    : dim(dim) {
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
    w1 = register_module("w1", torch::nn::Linear(dim, hidden_dim));
    w2 = register_module("w2", torch::nn::Linear(hidden_dim, dim));
    w3 = register_module("w3", torch::nn::Linear(dim, hidden_dim));
  }

  torch::Tensor forward(torch::Tensor x) {
    return w2(torch::silu(w1(x)) * w3(x));
  }

private:
  torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};
  int64_t dim, hidden_dim;
};
TORCH_MODULE(FeedForward);

class TransformerBlockImpl : public torch::nn::Module {
public:
  TransformerBlockImpl(int64_t dim, int64_t multiple_of, int64_t n_heads, double norm_eps)
    : attention(Attention(dim, n_heads)),
      feed_forward(FeedForward(dim, 4 * dim, multiple_of)),
      attention_norm(RMSNorm(dim, norm_eps)),
      ffn_norm(RMSNorm(dim, norm_eps)) {
    register_module("attention", attention);
    register_module("feed_forward", feed_forward);
    register_module("attention_norm", attention_norm);
    register_module("ffn_norm", ffn_norm);
  }

  torch::Tensor forward(torch::Tensor x, int64_t start_pos, torch::Tensor freqs_cis, torch::Tensor mask) {
    torch::Tensor h = x + attention->forward(attention_norm->forward(x), start_pos, freqs_cis, mask);
    return h + feed_forward->forward(ffn_norm->forward(h));
  }
private:
  Attention attention;
  FeedForward feed_forward;
  RMSNorm attention_norm, ffn_norm;
};
TORCH_MODULE(TransformerBlock);

class TransformerImpl : public torch::nn::Module {
public:
  TransformerImpl(int64_t dim, int64_t multiple_of, int64_t n_heads, int64_t n_layers, 
    double norm_eps, int64_t vocab_size, int64_t max_batch_size=32, int64_t max_seq_len=2048) 
    : dim(dim),
      multiple_of(multiple_of),
      n_heads(n_heads),
      n_layers(n_layers),
      norm_eps(norm_eps),
      max_batch_size(max_batch_size),
      max_seq_len(max_seq_len) {
    // for (int i = 0; i < n_layers; ++i) {
    //   layers.push_back(TransformerBlock(dim, multiple_of, n_heads, norm_eps));
    // }
    norm = register_module("norm", RMSNorm(dim, norm_eps));
    tok_embeddings = register_module("tok_embeddings", torch::nn::Embedding(vocab_size, dim));
    layer = register_module("block", TransformerBlock(dim, multiple_of, n_heads, norm_eps));
    output = register_module("output", torch::nn::Linear(dim, vocab_size));
    freqs_cis = precompute_freqs_cis(dim / n_heads, max_seq_len);
  }

  torch::Tensor forward(torch::Tensor tokens, int64_t start_pos, torch::Device device=torch::kCPU) {
    int64_t _bsz = tokens.sizes()[0];
    int64_t seqlen = tokens.sizes()[1];
    
    auto h = tok_embeddings->forward(tokens);
    auto freqs_cis_sliced = freqs_cis.slice(0, start_pos, start_pos + seqlen).to(device);

    torch::Tensor mask = torch::empty({1, 1, seqlen, seqlen}, device).fill_(std::numeric_limits<float>::lowest());
    mask.triu_(start_pos + 1);

    // for (int layer = 0; layer < n_layers; ++layer) {
    //   h = layers[layer]->forward(h, start_pos, freqs_cis_sliced, mask);
    // }
    h = layer->forward(h, start_pos, freqs_cis_sliced, mask);
    return output->forward((norm->forward(h)).narrow(1, -1, 1).squeeze(1)); // only compute the last logits
  }
private:
  int64_t dim, multiple_of, n_heads, n_layers, cache_size, max_batch_size, max_seq_len;
  double norm_eps;
  std::string param;
  // std::vector<TransformerBlock> layers;
  TransformerBlock layer{nullptr};
  RMSNorm norm{nullptr};
  torch::nn::Embedding tok_embeddings{nullptr};
  torch::nn::Linear output{nullptr};
  torch::Tensor freqs_cis;
};
TORCH_MODULE(Transformer);

int main() {
  torch::manual_seed(1337);
  const std::string PARAM = "7B";
  const int VOCAB_SIZE = 32000;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Device: CUDA" << std::endl;
    device = torch::kCUDA;
  }
  sentencepiece::SentencePieceProcessor processor;
  const auto status = processor.Load("../../weights/tokenizer.model");
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
  }

  Transformer model(512, 256, 8, 8, 1e-6, VOCAB_SIZE);
  model->to(device);
  // load state dict
  std::string prompt = "Elon Musk is ";

  std::vector<int> ids;
  processor.Encode(prompt, &ids);

  while (true) {
    torch::InferenceMode guard;
    torch::Tensor logits;
    torch::Tensor toks = torch::from_blob(ids.data(), {static_cast<int64_t>(ids.size())}, torch::kInt32);
    {
      torch::NoGradGuard no_grad;
      torch::Tensor input_tensor = toks.unsqueeze(0).to(device);
      logits = model(input_tensor, 0, device);
    }
    int tok = sample(logits, 0.7);
    ids.push_back(tok);

    std::string text;
    processor.Decode(ids, &text);
    std::cout << text << std::endl;
  }

  return 0;
}