class NeuralNetwork
    def initialize
        @w1 = rand
        @w2 = rand
        @b = rand
        @leaning_rate = 0.1
    end

    def sigmoid(x)
        1 / (1 + Math.exp(-x))
    end

    def sigmoid_derivative(x)
        x * (1 - x)
    end

    def predict(input1, input2)
        sigmoid(@w1 * input1 + @w2 * input2 + @b)
    end

    def train(training_data, labels, epochs = 10000)
        epochs.times do 
            training_data.each_with_index do |(x1, x2), index|
                y = labels[index]
                output = predict(x1, x2)

                error = y - output

                @w1 += @leaning_rate * error * sigmoid_derivative(output) * x1
                @w2 += @leaning_rate * error * sigmoid_derivative(output) * x2
                @b += @leaning_rate * error * sigmoid_derivative(output)
            end
        end
        p "Post train:"
        p @w1
        p @w2
        p @b
    end
end

inputs  = [[0,0], [0,1], [1,0], [1,1]]
outputs = [0, 0, 0, 1]

nn = NeuralNetwork.new
nn.train(inputs, outputs)

# Test predictions
inputs.each do |x1, x2|
  puts "#{x1}, #{x2} => #{nn.predict(x1, x2).round}"
end