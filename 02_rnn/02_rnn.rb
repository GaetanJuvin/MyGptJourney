srand = 1

def one_hot(i, n)
    v = Array.new(n, 0.0)
    v[i] = 1.0
    v
end

def softmax(v)
    m = v.max
    ex = v.map { |x| Math.exp(x - m) }
    s = ex.sum
    ex.map { |x| x / s }
end

def tanh(v)
    v.map { |x| Math.tanh(x) }
end

def dtanh(v)
    v.map { |y| 1.0 - y*y }
end

def matvec(m, v)
    m.map { |row| row.each_with_index.reduce(0.0) { |s, (w, j)| s + w * v[j] } }
end

def vecadd(a, b); a.each_with_index.map { |x, i| x + b[i] }; end
def vecsub(a, b); a.each_with_index.map { |x, i| x - b[i] }; end

def velem(a, b); a.each_with_index.map { |x, i| x * b[i] }; end
def vscale(a, s); a.map { |x| x * s }; end

def outer(a, b); a.map { |ai| b.map {|bj| ai * bj}}; end

def zeros(r, c=nil)
  if c.nil?
    Array.new(r, 0.0)
  else
    Array.new(r) { Array.new(c, 0.0) }
  end
end

def addmat!(a, b)
  a.each_index { |i| a[i].each_index { |j| a[i][j] += b[i][j] } }
end

def addvec!(a, b)
  a.each_index { |i| a[i] += b[i] }
end

def transpose(m)
  r = m.length
  c = m[0].length
  (0...c).map { |j| (0...r).map { |i| m[i][j] } }
end

def clip!(v, lo=-5.0, hi=5.0)
  v.each_index { |i| v[i] = [[v[i], lo].max, hi].min }
end

def clip_mat!(m, lo=-5.0, hi=5.0)
  m.each { |row| clip!(row, lo, hi) }
end

text = "hello world!"
data = text.chars
vocab = data.uniq.sort
vocab_size = vocab.size
char_to_ix = vocab.each_with_index.to_h
ix_to_char = vocab.each_with_index.map { |ch, i| [i, ch] }.to_h

inputs = data[0..-2].map { |ch| char_to_ix[ch] }
targets = data[1..-1].map { |ch| char_to_ix[ch] }

hidden_size = 32
seq_len = inputs.length
lr = 0.1

# Xavier-ish init
scale_x = Math.sqrt(1.0 / vocab_size)
scale_h = Math.sqrt(1.0 / hidden_size)

wxh = Array.new(hidden_size) { Array.new(vocab_size) { (rand*2-1) * scale_x } } # h <- x
whh = Array.new(hidden_size) { Array.new(hidden_size){ (rand*2-1) * scale_h } } # h <- h
why = Array.new(vocab_size)  { Array.new(hidden_size){ (rand*2-1) * scale_h } } # y <- h
bh  = zeros(hidden_size)
by  = zeros(vocab_size)

# def sample(ix_to_char, wxh, whh, why, bh, by, seed_ix, n, hidden_size)
#   h = zeros(hidden_size)
#   x = one_hot(seed_ix, wxh[0].length)
#   out = []
#   n.times do
#     h = tanh(vecadd(vecadd(matvec(wxh, x), matvec(whh, h)), bh))
#     y = vecadd(matvec(why, h), by)
#     p = softmax(y)
#     ix = p.each_with_index.max_by { |v, i| v }[1] # argmax sampling
#     out << ix_to_char[ix]
#     x = one_hot(ix, x.length)
#   end
#   out.join
# end

epochs = 400 # small dataset, many passes
epochs.times do |ep|
  # forward pass through full sequence
  hprev = zeros(hidden_size)
  xs, hs, ys, ps = [], [], [], []
  loss = 0.0

  seq_len.times do |t|
    x_t = one_hot(inputs[t], vocab_size)
    xs << x_t
    h_t = tanh(vecadd(vecadd(matvec(wxh, x_t), matvec(whh, hprev)), bh))
    hs << h_t
    y_t = vecadd(matvec(why, h_t), by)
    ys << y_t
    p_t = softmax(y_t)
    ps << p_t
    loss += -Math.log(p_t[targets[t]] + 1e-12)
    hprev = h_t
  end
  loss /= seq_len

  # backward pass
  dWxh = zeros(hidden_size, vocab_size)
  dWhh = zeros(hidden_size, hidden_size)
  dWhy = zeros(vocab_size, hidden_size)
  dbh  = zeros(hidden_size)
  dby  = zeros(vocab_size)
  dh_next = zeros(hidden_size)

  (seq_len-1).downto(0) do |t|
    dy = ps[t].dup
    dy[targets[t]] -= 1.0  # dL/dy
    addmat!(dWhy, outer(dy, hs[t]))
    addvec!(dby, dy)

    # backprop into h
    dh = vecadd(matvec(transpose(why), dy), dh_next)
    dhraw = velem(dtanh(hs[t]), dh)
    addmat!(dWxh, outer(dhraw, xs[t]))
    addmat!(dWhh, outer(dhraw, t > 0 ? hs[t-1] : zeros(hidden_size)))
    addvec!(dbh, dhraw)
    dh_next = matvec(transpose(whh), dhraw)
  end

  # clip gradients
  [dWxh, dWhh, dWhy].each { |m| clip_mat!(m) }
  [dbh, dby].each { |v| clip!(v) }

  # SGD update
  hidden_size.times do |i|
    vocab_size.times   { |j| wxh[i][j] -= lr * dWxh[i][j] }
    hidden_size.times  { |j| whh[i][j] -= lr * dWhh[i][j] }
  end
  vocab_size.times do |i|
    hidden_size.times  { |j| why[i][j] -= lr * dWhy[i][j] }
  end
  hidden_size.times   { |i| bh[i] -= lr * dbh[i] }
  vocab_size.times    { |i| by[i] -= lr * dby[i] }

#   if (ep+1) % 50 == 0
#     seed = char_to_ix[data[0]]
#     gen = sample(ix_to_char, wxh, whh, why, bh, by, seed, 20, hidden_size)
#     puts "Epoch #{ep+1}, loss=#{format('%.4f', loss)} | sample: #{gen.inspect}"
#   end
end

# puts "\nFinal samples:"
# seed = char_to_ix[data[0]]
# 3.times do
#   puts sample(ix_to_char, wxh, whh, why, bh, by, seed, 20, hidden_size)
# end


def predict_next_char(prefix, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
  h = zeros(hidden_size)
  vocab_size = wxh[0].length

  # run the prefix through the RNN to update hidden state
  prefix.chars.each do |ch|
    ix = char_to_ix.fetch(ch)
    x = one_hot(ix, vocab_size)
    h = tanh(vecadd(vecadd(matvec(wxh, x), matvec(whh, h)), bh))
  end

  # one more step to get next-char distribution
  y = vecadd(matvec(why, h), by)
  p = softmax(y)

  # top 5 candidates
  top = p.each_with_index.to_a.sort_by { |pair| -pair[0] }.take(5)
  {
    next_char: ix_to_char[top.first[1]],
    top5: top.map { |prob, idx| [ix_to_char[idx], prob] }
  }
end

def predict_next_char(prefix, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
  h = zeros(hidden_size)
  vocab_size = wxh[0].length

  # run the prefix through the RNN to update hidden state
  prefix.chars.each do |ch|
    ix = char_to_ix.fetch(ch)
    x = one_hot(ix, vocab_size)
    h = tanh(vecadd(vecadd(matvec(wxh, x), matvec(whh, h)), bh))
  end

  # one more step to get next-char distribution
  y = vecadd(matvec(why, h), by)
  p = softmax(y)

  # top 5 candidates
  top = p.each_with_index.to_a.sort_by { |pair| -pair[0] }.take(5)
  {
    next_char: ix_to_char[top.first[1]],
    top5: top.map { |prob, idx| [ix_to_char[idx], prob] }
  }
end

def generate_with_prefix(prefix, n, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
  h = zeros(hidden_size)
  vocab_size = wxh[0].length

  # feed the whole prefix
  last_ix = nil
  prefix.chars.each do |ch|
    last_ix = char_to_ix.fetch(ch)
    x = one_hot(last_ix, vocab_size)
    h = tanh(vecadd(vecadd(matvec(wxh, x), matvec(whh, h)), bh))
  end

  # now autoregress for n chars (argmax; swap for sampling if you like)
  out = prefix.dup
  n.times do
    y = vecadd(matvec(why, h), by)
    p = softmax(y)
    ix = p.each_with_index.max_by { |v, i| v }[1]
    out << ix_to_char[ix]
    x = one_hot(ix, vocab_size)
    h = tanh(vecadd(vecadd(matvec(wxh, x), matvec(whh, h)), bh))
  end
  out
end


# Example usage after training:
input = "hello wo"
res = predict_next_char(input, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
puts "Current: #{input} -> #{res[:next_char]}??"
puts "Top5: #{res[:top5].map { |ch, pr| "#{ch}=#{pr.round(3)}" }.join(', ')}"

input += res[:next_char]
res = predict_next_char(input, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
puts "Current: #{input} -> #{res[:next_char]}??"
puts "Top5: #{res[:top5].map { |ch, pr| "#{ch}=#{pr.round(3)}" }.join(', ')}"

input += res[:next_char]
res = predict_next_char(input, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
puts "Current: #{input} -> #{res[:next_char]}??"
puts "Top5: #{res[:top5].map { |ch, pr| "#{ch}=#{pr.round(3)}" }.join(', ')}"

input += res[:next_char]
res = predict_next_char(input, wxh, whh, why, bh, by, char_to_ix, ix_to_char, hidden_size)
puts "Current: #{input} -> #{res[:next_char]}??"
puts "Top5: #{res[:top5].map { |ch, pr| "#{ch}=#{pr.round(3)}" }.join(', ')}"
