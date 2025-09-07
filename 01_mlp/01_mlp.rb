srand = 42

class MLP
    def initialize(input_size: 2, hidden_size: 4, output_size: 1, lr: 0.5)
        @lr = lr

        scale_h = Math.sqrt(1.0 / input_size)
        scale_o = Math.sqrt(1.0 / hidden_size)

        @w1 = Array.new(hidden_size) { Array.new(input_size) { (rand * 2 - 1) * scale_h } }
        @b1 = Array.new(hidden_size, 0.0)
        @w2 = Array.new(output_size) { Array.new(hidden_size) { (rand * 2 - 1) * scale_o } }
        @b2 = Array.new(output_size, 0.0)
    end

    def sigmoid(x)
        1.0 / (1.0 + Math.exp(-x))
    end

    def dsigmoid(y)
        y * (1.0 - y)
    end

    def forward(x)
        z1 = @w1.map.with_index { |row, j| row[0] * x[0] + row[1] * x[1] + @b1[j] }
        h = z1.map { |v| sigmoid(v)}


        z2 = @w2.map.with_index { |row, k| row.each_with_index.sum { |w, j| w * h[j] } + @b2[k] }
        y = z2.map { |v| sigmoid(v) }

        [h, y]
    end

    def train_batch(data, labels, epochs: 5000, print_every: 500)
        epochs.times do |e|
            total_loss = 0.0

            data.zip(labels).each do |x, t|
                h, y = forward(x)

                loss = 0.5 * (t - y[0]) ** 2
                total_loss += loss

                dy = (y[0] - t) * dsigmoid(y[0])

                dw2 = h.map { |hj| dy * hj }
                db2 = dy

                dh = @w2[0].each_with_index.map { |w2j, j| w2j * dy }
                dz1 = dh.each_with_index.map { |val, j| val * dsigmoid(h[j]) }

                dw1 = @w1.map.with_index do |row, j|
                    [dz1[j] * x[0], dz1[j] * x[1]]
                end
                db1 = dz1

                @w2[0].each_index { |j| @w2[0][j] -= @lr * dw2[j] }
                @b2[0] -= @lr * db2

                @w1.each_index do |j|
                    @w1[j][0] -= @lr * dw1[j][0]
                    @w1[j][1] -= @lr * dw1[j][1]
                    @b1[j]    -= @lr * db1[j]
                end

                if (e+1) % print_every
                    puts "Epoch #{e + 1}, loss: #{format('%.6f', total_loss / data.size)}"
                end
            end
        end
    end

    def predict(x)
        _h, y = forward(x)
        y[0]
    end
end

# XOR dataset
data   = [[0,0], [0,1], [1,0], [1,1]]
labels = [0, 1, 1, 0]

net = MLP.new(input_size: 2, hidden_size: 4, output_size: 1, lr: 0.8)
net.train_batch(data, labels, epochs: 5000, print_every: 500)

puts "\nPredictions:"
data.each do |x|
  y = net.predict(x)
  puts "#{x.inspect} => #{y.round} (raw: #{format('%.4f', y)})"
end
