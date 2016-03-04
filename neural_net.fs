open FSharp.NeuralNet

[<EntryPoint>]
let main args =
  let layer_sizes = [264;15;10]
  let net = Network(layer_sizes)
  printfn "%A" net
  0
