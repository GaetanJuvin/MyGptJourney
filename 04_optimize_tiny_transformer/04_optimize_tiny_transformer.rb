begin
  require "torch-rb"
rescue LoadError
  require "torch"
end

unless Torch.const_defined?(:NN) && Torch.const_defined?(:Optim)
  raise "Torch NN and Optim modules not available. Please install torch-rb: gem install torch-rb"
end

puts "[Torch] loaded"

module Checkpointing
  CKPT_PATH = "tiny_transformer.ckpt"

  def self.save!(model, chars, path = CKPT_PATH)
    meta = {
      chars:   chars,
      d_model: model.d_model,
      max_len: model.max_len
    }
    File.binwrite(path, Marshal.dump({ meta: meta }))
    puts "[ckpt] saved to #{path}"
  end

  def self.load_meta(path = CKPT_PATH)
    payload = Marshal.load(File.binread(path))
    payload[:meta]
  end
end


def evaluate_tokens(model, tokens, block_size: 256)
  total_tokens = tokens.length
  return { loss: Float::NAN, acc: Float::NAN, n: 0 } if total_tokens < 2
  start_index = 0
  total_loss = 0.0
  total_correct = 0
  total_count = 0
  while (start_index + 1) < total_tokens
    end_index = [start_index + block_size, total_tokens - 1].min
    input_tokens = tokens[start_index...end_index]
    target_tokens = tokens[(start_index + 1)..end_index]
    cache = model.forward(input_tokens)
    probs = cache[:probs]
    sequence_length = probs.shape[0]
    effective_targets = target_tokens.last(sequence_length)
    loss = model.loss(probs, effective_targets)
    (0...sequence_length).each do |token_index|
      predicted_index = probs[token_index, true].flatten.max_index
      total_correct += 1 if predicted_index == effective_targets[token_index]
    end
    total_loss += loss * sequence_length
    total_count += sequence_length
    start_index = end_index
  end
  { loss: total_loss / total_count, acc: total_correct.to_f / total_count, n: total_count }
end

module NeuralFunctions
  def self.softmax(input_tensor, dim: 1)
    Torch.softmax(input_tensor, dim: dim)
  end
  
  def self.cross_entropy_loss(logits, target_indices)
    @cross_entropy_loss ||= Torch::NN::CrossEntropyLoss.new
    @cross_entropy_loss.call(logits, target_indices)
  end
end

class TinyTransformerTorch < Torch::NN::Module
  attr_reader :vocab_size, :d_model, :max_len
  attr_reader :E, :P, :W_Q, :W_K, :W_V, :W_O, :W_out, :b_out

  def initialize(vocab_size:, d_model: 64, max_len: 512, seed: 7)
    super()
    @vocab_size, @d_model, @max_len = vocab_size, d_model, max_len
    Torch.manual_seed(seed)
    scale = 0.02

    @E     = Torch::NN::Parameter.new(Torch.rand(vocab_size, d_model) * (2 * scale) - scale)
    @P     = Torch::NN::Parameter.new(Torch.rand(max_len,   d_model) * (2 * scale) - scale)
    @W_Q   = Torch::NN::Parameter.new(Torch.rand(d_model, d_model) * (2 * scale) - scale)
    @W_K   = Torch::NN::Parameter.new(Torch.rand(d_model, d_model) * (2 * scale) - scale)
    @W_V   = Torch::NN::Parameter.new(Torch.rand(d_model, d_model) * (2 * scale) - scale)
    @W_O   = Torch::NN::Parameter.new(Torch.rand(d_model, d_model) * (2 * scale) - scale)
    @W_out = Torch::NN::Parameter.new(Torch.rand(d_model, vocab_size) * (2 * scale) - scale)
    @b_out = Torch::NN::Parameter.new(Torch.zeros(vocab_size))
    @ln1   = Torch::NN::LayerNorm.new([d_model])
  end

  def embed_sequence(token_indices)
    token_indices = token_indices.last(@max_len)
    sequence_length = token_indices.length
    token_index_tensor = Torch.tensor(token_indices, dtype: :int64)
    position_index_tensor = Torch.arange(sequence_length, dtype: :int64)
    embedded_sequence = Torch.index_select(@E, 0, token_index_tensor) +
                        Torch.index_select(@P, 0, position_index_tensor)
    [embedded_sequence, token_indices]
  end

  def forward(token_indices)
    embedded_sequence, token_indices = embed_sequence(token_indices)
    queries = embedded_sequence.matmul(@W_Q)
    keys    = embedded_sequence.matmul(@W_K)
    values  = embedded_sequence.matmul(@W_V)
    attention_scores = queries.matmul(keys.transpose(0,1)) * (1.0 / Math.sqrt(@d_model))
    time_steps = attention_scores.size(0)
    causal_mask = Torch.ones([time_steps, time_steps]).triu(1)
    attention_scores = attention_scores + causal_mask * (-1e9)
    attention_weights = NeuralFunctions.softmax(attention_scores, dim: 1)
    attention_output  = attention_weights.matmul(values)
    projected_attention = attention_output.matmul(@W_O)
    residual_normalized = @ln1.call(embedded_sequence + projected_attention)
    logits = residual_normalized.matmul(@W_out) + @b_out
    logits
  end

  def step(prefix, char_to_ix, ix_to_char)
    known_characters = char_to_ix.keys
    sanitized_prefix = prefix.chars.select { |character| known_characters.include?(character) }.join
    raise "Prefix empty after sanitization" if sanitized_prefix.empty?
    token_indices = sanitized_prefix.chars.map { |character| char_to_ix[character] }
    logits = forward(token_indices)
    probabilities = NeuralFunctions.softmax(logits[-1, true], dim: 1).flatten
    predicted_index = probabilities.argmax.item
    [ix_to_char[predicted_index], probabilities]
  end

  def generate(prefix, max_length, char_to_ix, ix_to_char)
    output_text = prefix.dup
    max_length.times do
      next_character, _ = step(output_text, char_to_ix, ix_to_char)
      output_text << next_character
    end
    output_text
  end
end

class TorchWindowTrainer
  def initialize(model, tokens, lr: 0.0005, epochs: 400, steps_per_epoch: 200,
                 block_size: 192, print_every: 50, val_frac: 0.1, clip_norm: 1.0,
                 weight_decay: 0.0, warmup_steps: 200, plateau_patience: 50, plateau_factor: 0.5,
                 early_stop_patience: 150)
    @model = model
    @optimizer = Torch::Optim::Adam.new(@model.parameters, lr: lr, weight_decay: weight_decay)
    @epochs,@steps_per_epoch,@block_size,@print_every = epochs, steps_per_epoch, block_size, print_every
    @clip_norm = clip_norm
    @warmup_steps = warmup_steps
    @plateau_patience = plateau_patience
    @plateau_factor = plateau_factor
    @early_stop_patience = early_stop_patience
    @global_step = 0
    @best_val = Float::INFINITY
    @since_best = 0
    @base_lrs = @optimizer.param_groups.map { |param_group| param_group[:lr] }
    cutoff_index = [(tokens.length * (1.0 - val_frac)).floor, 2].max
    @train_tokens = tokens[0...cutoff_index]
    @validation_tokens   = tokens[cutoff_index..-1]
  end
  def sample_window
    max_start_index = [@train_tokens.length - (@block_size + 1), 0].max
    start_index = max_start_index.zero? ? 0 : rand(0..max_start_index)
    [@train_tokens[start_index, @block_size], @train_tokens[start_index+1, @block_size]]
  end

  def eval_tokens(tokens)
    return { loss: Float::NAN, acc: Float::NAN, n: 0 } if tokens.length < 2
    @model.eval
    tot_loss = 0.0
    tot_correct = 0
    tot = 0
    Torch.no_grad do
      start_index = 0
      while (start_index + 1) < tokens.length
        end_index = [start_index + @block_size, tokens.length - 1].min
        input_window = tokens[start_index...end_index]
        target_window = tokens[(start_index + 1)..end_index]
        logits = @model.forward(input_window)
        effective_length = logits.size(0)
        effective_targets = Torch.tensor(target_window.last(effective_length), dtype: :int64)
        loss = NeuralFunctions.cross_entropy_loss(logits, effective_targets)
        predictions = logits.argmax(1)
        tot_correct += (predictions.eq(effective_targets)).sum.item
        tot_loss += loss.item * effective_length
        tot += effective_length
        start_index = end_index
      end
    end
    @model.train
    { loss: tot_loss / tot, acc: tot_correct.to_f / tot, n: tot }
  end

  def train!
    last = nil
    @epochs.times do |epoch|
      p "epoch #{epoch}"
      sum = 0.0
      @steps_per_epoch.times do
        input_window, target_window = sample_window
        logits = @model.forward(input_window)
        effective_targets = Torch.tensor(target_window.last(logits.size(0)), dtype: :int64)
        loss = NeuralFunctions.cross_entropy_loss(logits, effective_targets)
        @optimizer.zero_grad
        loss.backward
        begin
          Torch::NN.utils.clip_grad_norm!(@model.parameters, @clip_norm)
        rescue NoMethodError
        end
        if @global_step < @warmup_steps
          warmup_scale = (@global_step + 1).to_f / @warmup_steps
          @optimizer.param_groups.each_with_index { |param_group, index| param_group[:lr] = @base_lrs[index] * warmup_scale }
        end
        @optimizer.step
        @global_step += 1
        sum += loss.item
      end
      train_loss = sum/@steps_per_epoch
      val = eval_tokens(@validation_tokens.empty? ? @train_tokens : @validation_tokens)
      if (epoch+1) % @print_every == 0
        cur_lr = @optimizer.param_groups.first[:lr]
        puts "Epoch #{epoch+1}, train_loss=#{format('%.4f', train_loss)} val_loss=#{format('%.4f', val[:loss])} val_acc=#{format('%.3f', val[:acc])} (n=#{val[:n]}) lr=#{format('%.2e', cur_lr)}"
      end
      if val[:loss] + 1e-6 < @best_val
        @best_val = val[:loss]
        @since_best = 0
      else
        @since_best += 1
        if @since_best == @plateau_patience
          @optimizer.param_groups.each { |param_group| param_group[:lr] = [param_group[:lr] * @plateau_factor, 1e-5].max }
        end
        if @since_best >= @early_stop_patience
          puts "Early stopping at epoch #{epoch+1} (no val improvement for #{@since_best} epochs)."
          return train_loss
        end
      end
      last = train_loss
    end
    last
  end
end



SYS = "<system>You are helpful.</system>\n"
SEP = "<eot>\n"
dialogs = [
  { user: "hi",            assistant: "hello!" },
  { user: "what is 2+2?",  assistant: "4" },
  { user: "color of sky?", assistant: "blue" }
]
one_pass = dialogs.map { |dialog|
  "#{SYS}<user>#{dialog[:user]}</user>\n<assistant>#{dialog[:assistant]}</assistant>\n#{SEP}"
}.join
train_text = one_pass * 300

if File.exist?(Checkpointing::CKPT_PATH)
  meta = Checkpointing.load_meta
  chars = meta[:chars]
  vocab_size = chars.length
  char_to_ix = chars.each_with_index.to_h
  ix_to_char = chars.each_with_index.map { |character, index| [index, character] }.to_h
  
  puts "Loading model from checkpoint"
  model = TinyTransformerTorch.new(
    vocab_size: vocab_size,
    d_model:    meta[:d_model],
    max_len:    meta[:max_len],
    seed:       7
  )
else
  chars = train_text.chars.uniq.sort
  vocab_size = chars.length
  char_to_ix = chars.each_with_index.to_h
  ix_to_char = chars.each_with_index.map { |character, index| [index, character] }.to_h

  tokens = train_text.chars.map { |character| char_to_ix[character] }

  puts "Training with Torch"
  model = TinyTransformerTorch.new(
    vocab_size: vocab_size,
    d_model:    64,
    max_len:    512,
    seed:       7
  )
  trainer = TorchWindowTrainer.new(
    model, tokens,
    lr: 0.0005, epochs: 400, steps_per_epoch: 200,
    block_size: 256, print_every: 1, clip_norm: 1.0
  )
  trainer.train!

  Checkpointing.save!(model, chars)
end

module Chat
  def self.build_prefix(user_text, system_prompt, known_characters)
    prefix = "#{system_prompt}<user>#{user_text}</user>\n<assistant>"
    filtered_prefix = prefix.chars.select { |character| known_characters.include?(character) }.join
    raise "input has no known characters for this vocab" if filtered_prefix.empty?
    filtered_prefix
  end

  def self.extract_reply(full_output, prefix, end_token)
    reply = full_output.sub(prefix, "")
    cutoff_index = reply.index(end_token) || reply.index("</assistant>")
    cutoff_index ? reply[0...cutoff_index] : reply
  end

  def self.start(model, system_prompt, end_token, char_to_ix, ix_to_char, max_length = 200)
    puts "\nChat ready. Type 'exit' to quit."
    loop do
      print "You> "
      user_input = STDIN.gets&.chomp
      break if user_input.nil? || user_input.strip.downcase == "exit"
      next if user_input.strip.empty?

      begin
        prefix = build_prefix(user_input, system_prompt, char_to_ix.keys)
        full_output = model.generate(prefix, max_length, char_to_ix, ix_to_char)
        reply = extract_reply(full_output, prefix, end_token)
        puts "Assistant> #{reply}"
      rescue => error
        puts "(warning) #{error.message}"
        next
      end
    end
  end
end

Chat.start(model, SYS, SEP, char_to_ix, ix_to_char)
