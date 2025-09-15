# tiny_transformer_class.rb
# Tiny Transformer in pure Ruby
# - Single attention head, one block
# - Token + positional embeddings
# - Causal mask
# - Cross entropy loss
# - Trainer class handles the loop

srand 7

# ============ helpers ============
def zeros(r, c=nil)
  if c.nil?
    Array.new(r, 0.0)
  else
    Array.new(r) { Array.new(c, 0.0) }
  end
end

def rand_mat(r, c, scale)
  Array.new(r) { Array.new(c) { (rand * 2 - 1) * scale } }
end

def matmul(a, b) # (m x n) * (n x p) -> (m x p)
  m = a.length
  n = a[0].length
  p = b[0].length
  out = Array.new(m) { Array.new(p, 0.0) }
  m.times do |i|
    p.times do |k|
      s = 0.0
      n.times { |j| s += a[i][j] * b[j][k] }
      out[i][k] = s
    end
  end
  out
end

def transpose(m)
  r = m.length
  c = m[0].length
  (0...c).map { |j| (0...r).map { |i| m[i][j] } }
end

def add_rows(a, b) # add vector b to each row of a
  a.map { |row| row.each_with_index.map { |x,i| x + b[i] } }
end

def add_mat!(a, b)
  a.each_index { |i| a[i].each_index { |j| a[i][j] += b[i][j] } }
end

def add_vec!(a, b)
  a.each_index { |i| a[i] += b[i] }
end

def softmax_rows(m)
  m.map do |row|
    mx = row.max
    ex = row.map { |x| Math.exp(x - mx) }
    s = ex.sum
    ex.map { |x| x / s }
  end
end

def softmax_jvp_row(a, da) # Jacobian-vector product for one softmax row
  dot = 0.0
  a.length.times { |i| dot += da[i] * a[i] }
  a.each_with_index.map { |ai,i| (da[i] - dot) * ai }
end

def causal_mask!(scores) # set j>i to large negative
  n = scores.length
  n.times do |i|
    n.times do |j|
      scores[i][j] = -1e9 if j > i
    end
  end
end

def sanitize_prefix(prefix, known)
  prefix.chars.select { |ch| known.include?(ch) }.join
end

# ============ Model ============
class TinyTransformer
  attr_reader :vocab_size, :d_model, :max_len
  attr_reader :E, :P, :W_Q, :W_K, :W_V, :W_O, :W_out, :b_out

  def initialize(vocab_size:, d_model: 16, max_len: 64, seed: 7)
    srand seed
    @vocab_size = vocab_size
    @d_model    = d_model
    @max_len    = max_len
    scale = Math.sqrt(1.0 / d_model)

    # parameters
    @E      = rand_mat(vocab_size, d_model, scale) # token embedding
    @P      = rand_mat(max_len,   d_model, scale)  # positional embedding
    @W_Q    = rand_mat(d_model, d_model, scale)
    @W_K    = rand_mat(d_model, d_model, scale)
    @W_V    = rand_mat(d_model, d_model, scale)
    @W_O    = rand_mat(d_model, d_model, scale)
    @W_out  = rand_mat(d_model, vocab_size, scale)
    @b_out  = zeros(vocab_size)
  end

  # ----- forward -----
  def embed_sequence(indices)
    x = Array.new(indices.length) { Array.new(@d_model, 0.0) }
    indices.each_with_index do |ix, t|
      @d_model.times { |j| x[t][j] = @E[ix][j] + @P[t][j] }
    end
    x
  end

  def forward_block(x)
    # x: (T x d)
    q = matmul(x, @W_Q) # (T x d)
    k = matmul(x, @W_K)
    v = matmul(x, @W_V)

    # scaled dot product attention
    kt = transpose(k)
    scores = matmul(q, kt)
    inv_sqrt = 1.0 / Math.sqrt(@d_model)
    scores.each_index { |i| scores[i].each_index { |j| scores[i][j] *= inv_sqrt } }

    causal_mask!(scores)
    a = softmax_rows(scores)  # (T x T)
    o = matmul(a, v)          # (T x d)
    h = matmul(o, @W_O)       # (T x d)
    [q, k, v, scores, a, o, h]
  end

  def forward(input_indices)
    x = embed_sequence(input_indices)                       # (T x d)
    q, k, v, scores, a, o, h = forward_block(x)
    logits = add_rows(matmul(h, @W_out), @b_out)           # (T x V)
    probs  = softmax_rows(logits)
    {
      x: x, q: q, k: k, v: v, scores: scores, a: a,
      o: o, h: h, logits: logits, probs: probs
    }
  end

  def loss(probs, targets)
    eps = 1e-12
    sum = 0.0
    probs.each_with_index { |row, t| sum += -Math.log(row[targets[t]] + eps) }
    sum / probs.length
  end

  # ----- backward, returns grads hash -----
  def backward(inputs, targets, cache)
    x      = cache[:x]
    q      = cache[:q]
    k      = cache[:k]
    v      = cache[:v]
    a      = cache[:a]
    o      = cache[:o]
    h      = cache[:h]
    probs  = cache[:probs]

    tcount = inputs.length

    dE      = zeros(@vocab_size, @d_model)
    dP      = zeros(@max_len, @d_model)
    dW_Q    = zeros(@d_model, @d_model)
    dW_K    = zeros(@d_model, @d_model)
    dW_V    = zeros(@d_model, @d_model)
    dW_O    = zeros(@d_model, @d_model)
    dW_out  = zeros(@d_model, @vocab_size)
    db_out  = zeros(@vocab_size)

    dx      = zeros(tcount, @d_model)
    dq      = zeros(tcount, @d_model)
    dk      = zeros(tcount, @d_model)
    dv      = zeros(tcount, @d_model)
    dscores = zeros(tcount, tcount)
    do_     = zeros(tcount, @d_model)
    dh      = zeros(tcount, @d_model)

    # d logits -> d h, dW_out, db_out
    probs.each_with_index do |row, t|
      dy = row.dup
      dy[targets[t]] -= 1.0
      @d_model.times do |i|
        @vocab_size.times do |j|
          dW_out[i][j] += h[t][i] * dy[j]
        end
      end
      @vocab_size.times { |j| db_out[j] += dy[j] }
      @d_model.times do |i|
        s = 0.0
        @vocab_size.times { |j| s += dy[j] * @W_out[i][j] }
        dh[t][i] += s
      end
    end

    # through W_O: h = o W_O
    tcount.times do |t|
      @d_model.times do |i|
        @d_model.times do |j|
          dW_O[i][j] += o[t][i] * dh[t][j]
        end
      end
      @d_model.times do |i|
        s = 0.0
        @d_model.times { |j| s += dh[t][j] * @W_O[i][j] }
        do_[t][i] += s
      end
    end

    # o = a v
    vt = transpose(v)
    at = transpose(a)
    da = matmul(do_, vt)     # (T x T)
    dv = matmul(at, do_)     # (T x d)

    # a = softmax(scores) rowwise
    tcount.times do |i|
      dscores[i] = softmax_jvp_row(a[i], da[i])
    end

    # scores = (q k^T) / sqrt(d)
    inv_sqrt = 1.0 / Math.sqrt(@d_model)
    dk_add = matmul(transpose(dscores), q) # (T x d)
    dq_add = matmul(dscores, k)            # (T x d)
    tcount.times do |t|
      @d_model.times do |j|
        dq[t][j] += dq_add[t][j] * inv_sqrt
        dk[t][j] += dk_add[t][j] * inv_sqrt
      end
    end

    # q = x W_Q ; k = x W_K ; v = x W_V
    xt = transpose(x)
    add_mat!(dW_Q, matmul(xt, dq))
    add_mat!(dW_K, matmul(xt, dk))
    add_mat!(dW_V, matmul(xt, dv))

    # dx += dq W_Q^T + dk W_K^T + dv W_V^T
    wq_t = transpose(@W_Q)
    wk_t = transpose(@W_K)
    wv_t = transpose(@W_V)
    add_mat!(dx, matmul(dq, wq_t))
    add_mat!(dx, matmul(dk, wk_t))
    add_mat!(dx, matmul(dv, wv_t))

    # back to embeddings
    inputs.each_with_index do |tok, t|
      @d_model.times do |j|
        dE[tok][j] += dx[t][j]
        dP[t][j]   += dx[t][j]
      end
    end

    # simple gradient clipping
    [dW_Q, dW_K, dW_V, dW_O, dW_out].each do |m|
      m.each do |row|
        row.each_index { |i| row[i] = [[row[i], -5.0].max, 5.0].min }
      end
    end
    db_out.each_index { |i| db_out[i] = [[db_out[i], -5.0].max, 5.0].min }

    {
      dE: dE, dP: dP, dW_Q: dW_Q, dW_K: dW_K, dW_V: dW_V,
      dW_O: dW_O, dW_out: dW_out, db_out: db_out
    }
  end

  # ----- update in place -----
  def update!(grads, lr)
    # embeddings
    @vocab_size.times do |i|
      @d_model.times do |j|
        @E[i][j]   -= lr * grads[:dE][i][j]
      end
    end
    @max_len.times do |i|
      @d_model.times do |j|
        @P[i][j]   -= lr * grads[:dP][i][j]
      end
    end
    # attention weights
    @d_model.times do |i|
      @d_model.times do |j|
        @W_Q[i][j] -= lr * grads[:dW_Q][i][j]
        @W_K[i][j] -= lr * grads[:dW_K][i][j]
        @W_V[i][j] -= lr * grads[:dW_V][i][j]
        @W_O[i][j] -= lr * grads[:dW_O][i][j]
      end
    end
    # classifier
    @d_model.times do |i|
      @vocab_size.times do |j|
        @W_out[i][j] -= lr * grads[:dW_out][i][j]
      end
    end
    @vocab_size.times { |i| @b_out[i] -= lr * grads[:db_out][i] }
  end

  # ----- inference -----
  def step(prefix, char_to_ix, ix_to_char)
    known = char_to_ix.keys
    pfx = sanitize_prefix(prefix, known)
    raise "Prefix empty after sanitization" if pfx.empty?
    idxs = pfx.chars.map { |ch| char_to_ix[ch] }
    cache = forward(idxs)
    p = cache[:probs][-1]
    ix = p.each_with_index.max_by { |v,i| v }[1]
    [ix_to_char[ix], p]
  end

  def generate(prefix, n, char_to_ix, ix_to_char)
    out = prefix.dup
    n.times do
      ch, _ = step(out, char_to_ix, ix_to_char)
      out << ch
    end
    out
  end
end

# ---- Optimizer: Adam ----
class Adam
  def initialize(lr: 0.003, beta1: 0.9, beta2: 0.999, eps: 1e-8)
    @lr, @b1, @b2, @eps = lr, beta1, beta2, eps
    @t = 0
    @state = {} # param_object_id => {m: same shape, v: same shape}
  end

  def _zeros_like(mat)
    mat[0].is_a?(Array) ? Array.new(mat.length) { Array.new(mat[0].length, 0.0) } : Array.new(mat.length, 0.0)
  end

  def _ensure_state_for(param)
    key = param.object_id
    @state[key] ||= { m: _zeros_like(param), v: _zeros_like(param) }
    @state[key]
  end

  def _adam_update!(param, grad, st)
    if param[0].is_a?(Array)
      param.length.times do |i|
        param[0].length.times do |j|
          st[:m][i][j] = @b1 * st[:m][i][j] + (1 - @b1) * grad[i][j]
          st[:v][i][j] = @b2 * st[:v][i][j] + (1 - @b2) * (grad[i][j] ** 2)
          mhat = st[:m][i][j] / (1 - @b1_pow)
          vhat = st[:v][i][j] / (1 - @b2_pow)
          param[i][j] -= @lr * mhat / (Math.sqrt(vhat) + @eps)
        end
      end
    else
      param.length.times do |i|
        st[:m][i] = @b1 * st[:m][i] + (1 - @b1) * grad[i]
        st[:v][i] = @b2 * st[:v][i] + (1 - @b2) * (grad[i] ** 2)
        mhat = st[:m][i] / (1 - @b1_pow)
        vhat = st[:v][i] / (1 - @b2_pow)
        param[i] -= @lr * mhat / (Math.sqrt(vhat) + @eps)
      end
    end
  end

  def step!(model, grads)
    @t += 1
    @b1_pow = @b1 ** @t
    @b2_pow = @b2 ** @t

    # update all parameters on the model with matching grads
    [
      [:E, :dE], [:P, :dP],
      [:W_Q, :dW_Q], [:W_K, :dW_K], [:W_V, :dW_V], [:W_O, :dW_O],
      [:W_out, :dW_out], [:b_out, :db_out]
    ].each do |param_sym, grad_sym|
      param = model.send(param_sym)
      grad  = grads[grad_sym]
      st    = _ensure_state_for(param)
      _adam_update!(param, grad, st)
    end
  end
end


# ============ Trainer ============
class Trainer
  def initialize(model, inputs, targets, lr: 0.003, epochs: 2000, print_every: 200)
    @model, @inputs, @targets = model, inputs, targets
    @epochs, @print_every = epochs, print_every
    @opt = Adam.new(lr: lr)
  end

  def train!
    last_loss = nil
    @epochs.times do |ep|
        p "epoch #{ep}"
        cache = @model.forward(@inputs)
        loss  = @model.loss(cache[:probs], @targets)
        grads = @model.backward(@inputs, @targets, cache)
        @opt.step!(@model, grads)

        if (ep + 1) % @print_every == 0
        puts "Epoch #{ep + 1}, loss=#{format('%.4f', loss)}"
        end
        last_loss = loss
    end
    last_loss
  end
end

# # ============ data ============
# text = "hello world!\n"
# chars = text.chars.uniq.sort
# vocab_size = chars.size
# char_to_ix = chars.each_with_index.to_h
# ix_to_char = chars.each_with_index.map { |ch, i| [i, ch] }.to_h

# tokens  = text.chars.map { |ch| char_to_ix[ch] }
# inputs  = tokens[0..-2]         # predict next char
# targets = tokens[1..-1]

# # ============ run ============
# model   = TinyTransformer.new(vocab_size: vocab_size, d_model: 16, max_len: 64, seed: 7)
# trainer = Trainer.new(model, inputs, targets, lr: 0.05, epochs: 450, print_every: 150)
# trainer.train!

# # puts "\nSamples:"
# # puts model.generate("hello worl", 5, char_to_ix, ix_to_char)

# txt_input = "hel"

# 8.times do
#     next_ch, probs = model.step(txt_input, char_to_ix, ix_to_char)
#     top5 = probs.each_with_index.sort_by { |v,i| -v }.take(5).map { |v,i| "#{ix_to_char[i]}=#{v.round(3)}" }.join(", ")
#     puts "#{txt_input} -> next(#{next_ch})"
#     puts "top5: #{top5}"
#     txt_input += next_ch
# end

dialogs = [
  {u: "hi",           a: "hello!"},
  {u: "what is 2+2?", a: "4"},
  {u: "color of sky?",a: "blue"},
]

SEP = "<eot>\n"
SYS = "<system>You are helpful.</system>\n"
train_text = dialogs.map { |d|
  "#{SYS}<user>#{d[:u]}</user>\n<assistant>#{d[:a]}</assistant>\n#{SEP}"
}.join

chars = train_text.chars.uniq.sort
vocab_size = chars.size
char_to_ix = chars.each_with_index.to_h
ix_to_char = chars.each_with_index.map { |ch,i| [i, ch] }.to_h

tokens  = train_text.chars.map { |ch| char_to_ix[ch] }
inputs  = tokens[0..-2]
targets = tokens[1..-1]

p "Transformer"
model   = TinyTransformer.new(vocab_size: vocab_size, d_model: 64, max_len: 512)
p "Trainer"
trainer = Trainer.new(model, inputs, targets, lr: 0.03, epochs: 3000, print_every: 200)
p "Trainer.train!"
trainer.train!

p "Trainer.generate"
puts model.generate("#{SYS}<user>hi</user>\n<assistant>", 40, char_to_ix, ix_to_char)