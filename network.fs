module FSharp.NeuralNet
  type Network(layer_sizes: int list) =
    let num_layers = layer_sizes.Length

    let layer_biases =
      List.tail layer_sizes
      |> List.map (fun x -> sprintf "Vector of length %i" x)

    let layer_weights =
      Seq.zip (Seq.take (num_layers - 1) layer_sizes) (Seq.skip 1 layer_sizes)
      |> Seq.map (fun (x, y) -> sprintf "Matrix of size %i,%i" y x)
      |> Seq.toList

    override this.ToString() = sprintf "Layer Sizes:%A\nLayer Biases: %A\nLayer Weights: %A" layer_sizes layer_biases layer_weights

  let sigmoid z = 1.0 / (1.0 + exp -z)
